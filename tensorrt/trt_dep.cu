#include "segmentation_cpp/SegmentationNode.hpp"
#include <fstream>
#include <opencv2/core.hpp>

using namespace std::chrono_literals;

inline void SegmentationNode::cudaCheck(cudaError_t e, const char* file, int line) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error %s:%d: %s\n", file, line, cudaGetErrorString(e));
        std::abort();
    }
}

SegmentationNode::SegmentationNode(const rclcpp::NodeOptions& options)
: Node("SegmentationNode", options), running_(true) {

    engine_path_     = declare_parameter<std::string>("engine_path", "segmentation.engine");
    input_width_     = declare_parameter<int>("input_width", 512);
    input_height_    = declare_parameter<int>("input_height", 288);
    input_channels_  = declare_parameter<int>("input_channels", 3);
    num_classes_     = declare_parameter<int>("num_classes", 2);
    batch_target_    = declare_parameter<int>("batch_target", 6);
    max_wait_ms_     = declare_parameter<int>("max_wait_ms", 3);
    threshold_       = declare_parameter<float>("mask_threshold", 0.5f);

    cam_topics_ = {
        declare_parameter<std::string>("cam0", "/cam0/image"),
        declare_parameter<std::string>("cam1", "/cam1/image"),
        declare_parameter<std::string>("cam2", "/cam2/image"),
        declare_parameter<std::string>("cam3", "/cam3/image"),
        declare_parameter<std::string>("cam4", "/cam4/image"),
        declare_parameter<std::string>("cam5", "/cam5/image")
    };
    out_topics_ = {
        declare_parameter<std::string>("out0", "/cam0/mask"),
        declare_parameter<std::string>("out1", "/cam1/mask"),
        declare_parameter<std::string>("out2", "/cam2/mask"),
        declare_parameter<std::string>("out3", "/cam3/mask"),
        declare_parameter<std::string>("out4", "/cam4/mask"),
        declare_parameter<std::string>("out5", "/cam5/mask")
    };

    init_trt();

    rclcpp::QoS qos = rclcpp::SensorDataQoS().keep_last(5).best_effort();
    for (size_t i = 0; i < kCams; ++i) {
        publishers_[i] = this->create_publisher<sensor_msgs::msg::Image>(out_topics_[i], qos);
        subscriptions_[i] = this->create_subscription<sensor_msgs::msg::Image>(
            cam_topics_[i], qos,
            [this, idx=i](sensor_msgs::msg::Image::ConstSharedPtr msg) {
                on_image(idx, msg);
            });
    }

    worker_ = std::thread([this]() { scheduler_loop(); });

    RCLCPP_INFO(get_logger(), "SegmentationNode ready: %dx%d, batch=%d", input_width_, input_height_, batch_target_);
}

SegmentationNode::~SegmentationNode() {
    running_ = false;
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
    free_buffers();
}

void SegmentationNode::on_image(size_t cam_idx, const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
    if (!msg || msg->data.empty()) return;
    {
        std::lock_guard<std::mutex> lk(q_mtx_[cam_idx]);
        q_[cam_idx].push_back(Sample{cam_idx, msg->header.stamp, msg});
    }
    cv_.notify_one();
}

void SegmentationNode::scheduler_loop() {
    while (running_) {
        {
            std::unique_lock<std::mutex> lk(cv_mtx_);
            cv_.wait_for(lk, std::chrono::milliseconds(max_wait_ms_), [this]() {
                for (auto& dq : q_) if (!dq.empty()) return true;
                return !running_;
            });
        }
        if (!running_) break;

        std::vector<Sample> batch;
        batch.reserve(batch_target_);
        for (size_t pass = 0; pass < kCams && batch.size() < (size_t)batch_target_; ++pass) {
            for (size_t i = 0; i < kCams && batch.size() < (size_t)batch_target_; ++i) {
                std::lock_guard<std::mutex> lk(q_mtx_[i]);
                if (!q_[i].empty()) {
                    batch.push_back(std::move(q_[i].front()));
                    q_[i].pop_front();
                }
            }
        }
        if (!batch.empty()) run_inference(batch);
    }
}

void SegmentationNode::run_inference(const std::vector<Sample>& batch) {
    int N = batch.size();
    nvinfer1::Dims4 in_shape(N, input_channels_, input_height_, input_width_);
    context_->setInputShape(input_tensor_.c_str(), in_shape);

    // Host staging
    std::vector<float> host_input(N * input_channels_ * input_height_ * input_width_);
    for (int i = 0; i < N; ++i) {
        const auto& msg = batch[i].msg;
        const uint8_t* src = msg->data.data();
        int ch = input_channels_;
        int src_step = msg->step;
        float* dst_ptr = host_input.data() + i * ch * input_height_ * input_width_;
        for (int y = 0; y < input_height_; ++y) {
            const uint8_t* row = src + y * src_step;
            for (int x = 0; x < input_width_; ++x) {
                for (int c = 0; c < ch; ++c) {
                    dst_ptr[(c * input_height_ + y) * input_width_ + x] =
                        row[x * ch + c] / 255.0f;
                }
            }
        }
    }

    // Copy to device
    size_t in_bytes = host_input.size() * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(d_input_, host_input.data(), in_bytes, cudaMemcpyHostToDevice, stream_));

    // Set addresses and run
    context_->setInputTensorAddress(input_tensor_.c_str(), d_input_);
    context_->setOutputTensorAddress(output_tensor_.c_str(), d_output_);
    context_->enqueueV3(stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    // Output shape
    nvinfer1::Dims out_dims = context_->getTensorShape(output_tensor_.c_str());
    int outC = out_dims.d[1], outH = out_dims.d[2], outW = out_dims.d[3];
    size_t outElems = N * outC * outH * outW;

    std::vector<float> logits(outElems);
    CUDA_CHECK(cudaMemcpy(logits.data(), d_output_, outElems * sizeof(float), cudaMemcpyDeviceToHost));

    // Argmax and publish
    for (int b = 0; b < N; ++b) {
        std::vector<uint8_t> labels(outH * outW);
        for (int h = 0; h < outH; ++h) {
            for (int w = 0; w < outW; ++w) {
                int best = 0;
                float mv = logits[(b * outC + 0) * outH * outW + h * outW + w];
                for (int c = 1; c < outC; ++c) {
                    float v = logits[(b * outC + c) * outH * outW + h * outW + w];
                    if (v > mv) { mv = v; best = c; }
                }
                labels[h * outW + w] = (best == 1 ? 255 : 0);
            }
        }
        auto msg = std::make_unique<sensor_msgs::msg::Image>();
        msg->header.stamp = batch[b].stamp;
        msg->header.frame_id = "map";
        msg->height = outH;
        msg->width  = outW;
        msg->encoding = "mono8";
        msg->is_bigendian = 0;
        msg->step = outW;
        msg->data = std::move(labels);
        publishers_[batch[b].cam_idx]->publish(std::move(msg));
    }
}

void SegmentationNode::init_trt() {
    // Load engine
    std::ifstream f(engine_path_, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open engine file");
    f.seekg(0, std::ios::end);
    size_t sz = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> blob(sz);
    f.read(blob.data(), sz);

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
    context_.reset(engine_->createExecutionContext());

    // Find tensor names
    int nIO = engine_->getNbIOTensors();
    for (int i = 0; i < nIO; ++i) {
        const char* name = engine_->getIOTensorName(i);
        auto mode = engine_->getTensorIOMode(name);
        if (mode == nvinfer1::TensorIOMode::kINPUT && input_tensor_.empty())
            input_tensor_ = name;
        else if (mode == nvinfer1::TensorIOMode::kOUTPUT && output_tensor_.empty())
            output_tensor_ = name;
    }

    // Allocate buffers
    max_batch_ = std::max(batch_target_, 1);
    size_t in_bytes = max_batch_ * input_channels_ * input_height_ * input_width_ * sizeof(float);
    // Assume FP32 output for allocation
    nvinfer1::Dims out_dims = engine_->getTensorShape(output_tensor_.c_str());
    int outC = num_classes_ > 1 ? num_classes_ : 1;
    int outH = (out_dims.d[2] > 0) ? out_dims.d[2] : input_height_;
    int outW = (out_dims.d[3] > 0) ? out_dims.d[3] : input_width_;
    size_t out_bytes = max_batch_ * outC * outH * outW * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_input_, in_bytes));
    CUDA_CHECK(cudaMalloc(&d_output_, out_bytes));
    CUDA_CHECK(cudaStreamCreate(&stream_));
}

void SegmentationNode::free_buffers() {
    if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
    if (d_input_) { cudaFree(d_input_); d_input_ = nullptr; }
    if (d_output_) { cudaFree(d_output_); d_output_ = nullptr; }
}

nvinfer1::ILogger* SegmentationNode::get_trt_logger() {
    return &logger_;
}
