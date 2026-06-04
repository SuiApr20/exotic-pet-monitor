/**
 * 异宠智能监护系统 — Vue 3 主应用
 */
const { createApp, ref, reactive, computed, watch, onMounted, nextTick } = Vue;

const app = createApp({
    setup() {
        // ============== 品种识别相关 ==============
        const fileInput = ref(null);
        const isDragging = ref(false);
        const uploadPreview = ref(null);
        const detecting = ref(false);
        const detectResult = ref(null);
        const selectedSpecies = ref("guinea_pig");
        const selectedColor = ref("");
        const selectedBreed = ref("");
        const monitorSpecies = ref("guinea_pig");

        const speciesList = {
            guinea_pig: "荷兰猪",
            parrot: "鹦鹉",
            hamster: "仓鼠"
        };

        const breedOptions = {
            colors: ["白色", "雕灰", "奶黄", "棕色", "三花"],
            breeds: ["长顺", "长逆", "加州", "泰迪"]
        };

        // 取置信度最高的检测结果
        const topDetection = computed(() => {
            if (!detectResult.value || !detectResult.value.detections || !detectResult.value.detections.length) {
                return { class_name: '未知', confidence: 0, breed_info: null };
            }
            const sorted = [...detectResult.value.detections].sort((a, b) => b.confidence - a.confidence);
            return sorted[0];
        });

        function getSpeciesIcon(key) {
            const icons = { guinea_pig: "🐾", parrot: "🦜", hamster: "🐹" };
            return icons[key] || "";
        }

        // ============== 健康监护相关 ==============
        const videoPlayer = ref(null);
        const videoInput = ref(null);
        const monitorCanvas = ref(null);
        const monitorActive = ref(false);
        const videoUploading = ref(false);
        const videoResult = ref(null);

        // ROI 框选
        const roiMode = ref(null);  // 'water_bottle' | 'hay_rack' | null
        const roiZones = ref([]);
        const roiDrawing = reactive({ active: false, startX: 0, startY: 0, endX: 0, endY: 0 });

        // 行为统计
        const behaviorStats = reactive({
            total_eating_sec: 0,
            total_drinking_sec: 0,
            eating_events: 0,
            drinking_events: 0,
        });

        const healthResult = ref(null);
        const behaviorChart = ref(null);
        let chartInstance = null;
        let wsConnection = null;
        let webcamStream = null;
        let monitorInterval = null;

        // ============== 品种识别方法 ==============
        function handleFileSelect(e) {
            const file = e.target.files[0];
            if (!file) return;
            showPreview(file);
        }

        function handleDrop(e) {
            isDragging.value = false;
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith("image/")) {
                showPreview(file);
            }
        }

        function showPreview(file) {
            const reader = new FileReader();
            reader.onload = (ev) => {
                uploadPreview.value = ev.target.result;
                detectResult.value = null;
            };
            reader.readAsDataURL(file);
        }

        async function detectBreed() {
            if (!uploadPreview.value) return;
            detecting.value = true;

            try {
                const formData = new FormData();
                const blob = dataURLtoBlob(uploadPreview.value);
                formData.append("file", blob, "upload.jpg");
                formData.append("species", selectedSpecies.value);

                const resp = await fetch("/api/upload", { method: "POST", body: formData });
                const data = await resp.json();

                if (data.success) {
                    detectResult.value = {
                        ...data,
                        result_image_url: data.result_image
                            ? "/uploads/" + data.result_image
                            : uploadPreview.value
                    };
                }
            } catch (err) {
                console.error("检测失败:", err);
                alert("检测失败: " + err.message);
            } finally {
                detecting.value = false;
            }
        }

        function dataURLtoBlob(dataURL) {
            const parts = dataURL.split(",");
            const mime = parts[0].match(/:(.*?);/)[1];
            const bytes = atob(parts[1]);
            const arr = new Uint8Array(bytes.length);
            for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i);
            return new Blob([arr], { type: mime });
        }

        // ============== 健康监护方法 ==============

        // -- 摄像头 --
        async function startWebcam() {
            try {
                webcamStream = await navigator.mediaDevices.getUserMedia({
                    video: { width: 640, height: 480, facingMode: "environment" }
                });
                videoPlayer.value.srcObject = webcamStream;
                videoPlayer.value.style.display = "block";
                monitorActive.value = true;
                videoResult.value = null;
                await connectWebSocket();
                processWebcamFrames();
            } catch (err) {
                alert("无法访问摄像头: " + err.message);
            }
        }

        function stopWebcam() {
            if (webcamStream) {
                webcamStream.getTracks().forEach(t => t.stop());
                webcamStream = null;
            }
            if (wsConnection) {
                wsConnection.close();
                wsConnection = null;
            }
            if (monitorInterval) {
                clearInterval(monitorInterval);
                monitorInterval = null;
            }
            monitorActive.value = false;
            videoPlayer.value.style.display = "none";
            drawEmptyCanvas();
        }

        function drawEmptyCanvas() {
            const canvas = monitorCanvas.value;
            if (!canvas) return;
            canvas.width = canvas.clientWidth;
            canvas.height = 400;
            const ctx = canvas.getContext("2d");
            ctx.fillStyle = "#1a1a2e";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            drawROIOnCanvas(ctx);
        }

        function processWebcamFrames() {
            const canvas = document.createElement("canvas");
            const video = videoPlayer.value;
            monitorInterval = setInterval(() => {
                if (!monitorActive.value || !wsConnection || wsConnection.readyState !== WebSocket.OPEN) return;
                canvas.width = video.videoWidth || 640;
                canvas.height = video.videoHeight || 480;
                const ctx = canvas.getContext("2d");
                ctx.drawImage(video, 0, 0);
                const frameBase64 = canvas.toDataURL("image/jpeg", 0.6);
                wsConnection.send(JSON.stringify({
                    frame: frameBase64,
                    roi_zones: roiZones.value
                }));
                // 同时在本机画布上显示视频
                const monCanvas = monitorCanvas.value;
                if (monCanvas) {
                    monCanvas.width = monCanvas.clientWidth;
                    monCanvas.height = monCanvas.clientHeight || 400;
                    const monCtx = monCanvas.getContext("2d");
                    monCtx.drawImage(video, 0, 0, monCanvas.width, monCanvas.height);
                    drawROIOnCanvas(monCtx);
                }
            }, 500); // 每秒2帧
        }

        async function connectWebSocket() {
            const protocol = location.protocol === "https:" ? "wss:" : "ws:";
            const wsUrl = `${protocol}//${location.host}/api/ws/monitor`;
            wsConnection = new WebSocket(wsUrl);

            wsConnection.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.success && data.result_image_base64) {
                    // 在画布上绘制返回的检测结果
                    const monCanvas = monitorCanvas.value;
                    if (!monCanvas) return;
                    const ctx = monCanvas.getContext("2d");
                    const img = new Image();
                    img.onload = () => {
                        monCanvas.width = monCanvas.clientWidth;
                        monCanvas.height = monCanvas.clientHeight || 400;
                        ctx.drawImage(img, 0, 0, monCanvas.width, monCanvas.height);
                        drawROIOnCanvas(ctx);
                    };
                    img.src = "data:image/jpeg;base64," + data.result_image_base64;
                }
                if (data.behavior) {
                    Object.assign(behaviorStats, data.behavior.stats_snapshot || {});
                }
            };

            wsConnection.onerror = (err) => console.error("WebSocket error:", err);
        }

        // -- ROI 绘制 --
        function canvasMouseDown(e) {
            if (!roiMode.value) return;
            const rect = monitorCanvas.value.getBoundingClientRect();
            roiDrawing.active = true;
            roiDrawing.startX = e.clientX - rect.left;
            roiDrawing.startY = e.clientY - rect.top;
            roiDrawing.endX = roiDrawing.startX;
            roiDrawing.endY = roiDrawing.startY;
        }

        function canvasMouseMove(e) {
            if (!roiDrawing.active) return;
            const rect = monitorCanvas.value.getBoundingClientRect();
            roiDrawing.endX = e.clientX - rect.left;
            roiDrawing.endY = e.clientY - rect.top;
            redrawCanvasWithROI();
        }

        function canvasMouseUp(e) {
            if (!roiDrawing.active) return;
            roiDrawing.active = false;
            const rect = monitorCanvas.value.getBoundingClientRect();

            const x1 = Math.min(roiDrawing.startX, roiDrawing.endX);
            const y1 = Math.min(roiDrawing.startY, roiDrawing.endY);
            const x2 = Math.max(roiDrawing.startX, roiDrawing.endX);
            const y2 = Math.max(roiDrawing.startY, roiDrawing.endY);

            const w = x2 - x1;
            const h = y2 - y1;

            if (w < 10 || h < 10) return; // 太小忽略

            // 移除同类型的旧ROI
            roiZones.value = roiZones.value.filter(z => z.name !== roiMode.value);

            roiZones.value.push({
                name: roiMode.value,
                x: Math.round(x1),
                y: Math.round(y1),
                width: Math.round(w),
                height: Math.round(h)
            });

            roiMode.value = null; // 退出框选模式
            redrawCanvasWithROI();
        }

        function redrawCanvasWithROI() {
            const canvas = monitorCanvas.value;
            if (!canvas) return;
            const ctx = canvas.getContext("2d");
            // 这个函数被 drawBox 调用时图像已在上面，只需覆盖 ROI
            drawROIOnCanvas(ctx);
            if (roiDrawing.active) {
                ctx.strokeStyle = "#ff0";
                ctx.lineWidth = 2;
                ctx.setLineDash([5, 3]);
                ctx.strokeRect(
                    roiDrawing.startX, roiDrawing.startY,
                    roiDrawing.endX - roiDrawing.startX,
                    roiDrawing.endY - roiDrawing.startY
                );
                ctx.setLineDash([]);
            }
        }

        function drawROIOnCanvas(ctx) {
            roiZones.value.forEach(z => {
                ctx.strokeStyle = z.name === "water_bottle" ? "#0d6efd" : "#198754";
                ctx.lineWidth = 3;
                ctx.setLineDash([]);
                ctx.strokeRect(z.x, z.y, z.width, z.height);
                ctx.fillStyle = z.name === "water_bottle"
                    ? "rgba(13,110,253,0.15)"
                    : "rgba(25,135,84,0.15)";
                ctx.fillRect(z.x, z.y, z.width, z.height);
                // 标签
                const label = z.name === "water_bottle" ? "💧 水瓶" : "🥬 草架";
                ctx.font = "14px sans-serif";
                ctx.fillStyle = z.name === "water_bottle" ? "#0d6efd" : "#198754";
                ctx.fillText(label, z.x, Math.max(z.y - 5, 15));
            });
        }

        function clearROI() {
            roiZones.value = [];
            roiMode.value = null;
            drawEmptyCanvas();
        }

        // -- 上传视频 --
        async function uploadVideo(e) {
            const file = e.target.files[0];
            if (!file) return;
            videoUploading.value = true;
            videoResult.value = null;

            try {
                const formData = new FormData();
                formData.append("file", file);
                formData.append("sample_rate", "2");

                const resp = await fetch("/api/video/analyze", { method: "POST", body: formData });
                const data = await resp.json();

                if (data.success) {
                    videoResult.value = data;
                    Object.assign(behaviorStats, data.stats || {});
                    healthResult.value = data.health;
                    updateChart(data.timeline || []);
                }
            } catch (err) {
                console.error("视频分析失败:", err);
                alert("视频分析失败: " + err.message);
            } finally {
                videoUploading.value = false;
            }
        }

        // -- 图表 --
        function initChart() {
            if (!behaviorChart.value) return;
            chartInstance = echarts.init(behaviorChart.value);
            chartInstance.setOption({
                title: { text: "行为时间线", textStyle: { fontSize: 12, color: "#666" } },
                tooltip: { trigger: "axis" },
                xAxis: { type: "value", name: "秒", axisLabel: { fontSize: 10 } },
                yAxis: { type: "category", data: ["饮水", "进食"], axisLabel: { fontSize: 10 } },
                series: [{
                    type: "scatter",
                    symbolSize: 10,
                    data: [],
                    itemStyle: { color: "#0d6efd" },
                    encode: { x: 0, y: 1 }
                }]
            });
        }

        function updateChart(timeline) {
            if (!chartInstance) initChart();
            if (!chartInstance) return;

            const eatingData = timeline
                .filter(t => t.type === "eating")
                .map(t => [t.time_sec, "进食"]);
            const drinkingData = timeline
                .filter(t => t.type === "drinking")
                .map(t => [t.time_sec, "饮水"]);

            chartInstance.setOption({
                series: [
                    {
                        type: "scatter",
                        symbolSize: (val) => Math.max(6, Math.min(val[2] / 3 || 8, 30)),
                        data: drinkingData,
                        itemStyle: { color: "#0d6efd" },
                        encode: { x: 0, y: 1 }
                    },
                    {
                        type: "scatter",
                        symbolSize: (val) => Math.max(6, Math.min(val[2] / 3 || 8, 30)),
                        data: eatingData,
                        itemStyle: { color: "#198754" },
                        encode: { x: 0, y: 1 }
                    }
                ]
            });
        }

        // ============== 生命周期 ==============
        onMounted(() => {
            drawEmptyCanvas();
        });

        // 监听 BMI Tab 切换以初始化图表
        const tabMonitorBtn = document.querySelector('[data-bs-target="#tab-monitor"]');
        if (tabMonitorBtn) {
            tabMonitorBtn.addEventListener("shown.bs.tab", () => {
                if (!chartInstance) {
                    nextTick(() => initChart());
                }
            });
        }

        return {
            // 品种识别
            fileInput, isDragging, uploadPreview, detecting, detectResult,
            selectedSpecies, selectedColor, selectedBreed,
            speciesList, breedOptions,
            monitorSpecies,
            topDetection,
            getSpeciesIcon, handleFileSelect, handleDrop, detectBreed,

            // 健康监护
            videoPlayer, videoInput, monitorCanvas,
            monitorActive, videoUploading, videoResult,
            roiMode, roiZones, roiDrawing,
            behaviorStats, healthResult, behaviorChart,

            startWebcam, stopWebcam,
            canvasMouseDown, canvasMouseMove, canvasMouseUp,
            clearROI, uploadVideo,
        };
    }
});

// 全局错误处理
app.config.errorHandler = (err, vm, info) => {
    console.error("Vue Error:", err, info);
};

app.mount("#app");
