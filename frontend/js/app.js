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
        const confThreshold = ref(0.25);  // 检测置信度阈值
        const monitorSpecies = ref("guinea_pig");

        // 模型切换 — 品种识别Tab
        const selectedModel = ref("best");
        const availableModels = ref([]);
        // 品种识别只显示非监护专用模型
        const breedModels = computed(() =>
            availableModels.value.filter(m => !m.for_monitor)
        );
        const currentModelInfo = computed(() => {
            return availableModels.value.find(m => m.key === selectedModel.value) || null;
        });

        // 健康监护Tab — 固定使用行为监测模型
        const monitorModelKey = ref("behavior");
        const behaviorAvailable = computed(() => {
            return availableModels.value.some(m => m.key === "behavior" && m.is_available);
        });

        const speciesList = {
            guinea_pig: "荷兰猪",
            parrot: "鹦鹉",
            hamster: "仓鼠"
        };

        const breedOptions = {
            colors: ["奶黄", "白色", "三花"],
            breeds: ["加州", "美泰"]
        };

        function modelLabel(key) {
            const m = availableModels.value.find(m => m.key === key);
            return m ? m.name : key;
        }

        function modelBadgeClass(key) {
            const map = { best: 'bg-primary', guinea_pig: 'bg-success', parrot: 'bg-warning text-dark', behavior: 'bg-info text-dark' };
            return map[key] || 'bg-secondary';
        }

        function modelBorderColor(key) {
            const map = { best: 'primary', guinea_pig: 'success', parrot: 'warning', behavior: 'info' };
            return map[key] || 'secondary';
        }

        function modelAlertClass(key) {
            const map = { best: 'alert-primary', guinea_pig: 'alert-success', parrot: 'alert-warning', behavior: 'alert-info' };
            return map[key] || 'alert-info';
        }

        function modelDetectionLabel(key) {
            const map = {
                best: '通用模型 - 三物种检测',
                guinea_pig: '荷兰猪专精 - 品种识别',
                parrot: '鹦鹉专精 - 物种检测',
                behavior: '行为监测 - 进食/饮水/品种'
            };
            return map[key] || '模型检测';
        }

        function getSpeciesIcon(key) {
            return '';  // deprecated, kept for compatibility
        }

        async function loadModels() {
            try {
                const resp = await fetch("/api/models");
                const data = await resp.json();
                availableModels.value = data.models || [];
                selectedModel.value = "best";
            } catch (err) {
                console.error("加载模型列表失败:", err);
            }
        }

        // 模型 ↔ 物种双向映射
        const modelSpeciesMap = { best: "guinea_pig", guinea_pig: "guinea_pig", parrot: "parrot" };
        const modelToSpecies = (modelKey) => modelSpeciesMap[modelKey] || "guinea_pig";

        async function onModelChange() {
            // 切模型时自动同步物种选择，清除旧结果
            detectResult.value = null;
            selectedSpecies.value = modelSpeciesMap[selectedModel.value] || "guinea_pig";
            try {
                const formData = new FormData();
                formData.append("model_key", selectedModel.value);
                await fetch("/api/switch-model", { method: "POST", body: formData });
            } catch (err) {
                console.error("模型切换失败:", err);
            }
        }

        // 取置信度最高的检测结果
        const topDetection = computed(() => {
            if (!detectResult.value || !detectResult.value.detections || !detectResult.value.detections.length) {
                return { class_name: '未知', confidence: 0, breed_info: null };
            }
            const sorted = [...detectResult.value.detections].sort((a, b) => b.confidence - a.confidence);
            return sorted[0];
        });

        // 结果展示名：专精模型直接显示品种名，通用模型显示物种名
        const detectionDisplayName = computed(() => {
            const td = topDetection.value;
            const model = detectResult.value?.model_used || 'best';
            if (model === 'guinea_pig' && td.breed_info && td.breed_info.full_name !== '未知') {
                return td.breed_info.color + td.breed_info.breed;
            }
            return td.class_name;
        });

        // 只在通用模型+检测到荷兰猪+有品种信息时显示品种卡片
        const showBreedCard = computed(() => {
            const td = topDetection.value;
            const model = detectResult.value?.model_used || 'best';
            return model === 'best'
                && td.class_name === '荷兰猪'
                && td.breed_info
                && td.breed_info.full_name !== '未知';
        });

        // ============== 健康监护相关 ==============
        const videoPlayer = ref(null);
        const reviewVideo = ref(null);
        const annotatedCanvas = ref(null);
        const videoInput = ref(null);
        const monitorCanvas = ref(null);
        const monitorActive = ref(false);
        const videoUploading = ref(false);
        const videoResult = ref(null);

        // 视频播放器相关
        const videoProgress = ref(0);
        const videoCurrentTime = ref(0);
        const videoTimeline = ref([]);
        const videoPlaying = ref(false);
        const frameDetections = ref([]);  // 后端返回的逐帧检测数据 [{time_sec, detections}]
        let animFrameId = null;

        function formatTime(sec) {
            const m = Math.floor(sec / 60), s = Math.floor(sec % 60);
            return m + ':' + String(s).padStart(2, '0');
        }

        function toggleVideoPlay() {
            const v = reviewVideo.value;
            if (!v) return;
            if (v.paused) {
                v.play();
                videoPlaying.value = true;
                startCanvasLoop();
            } else {
                v.pause();
                videoPlaying.value = false;
                stopCanvasLoop();
            }
        }

        function startCanvasLoop() {
            if (animFrameId) return;
            function loop() {
                drawAnnotatedFrame();
                animFrameId = requestAnimationFrame(loop);
            }
            loop();
        }

        function stopCanvasLoop() {
            if (animFrameId) {
                cancelAnimationFrame(animFrameId);
                animFrameId = null;
            }
            // 最后一帧可能已暂停，补画一次
            drawAnnotatedFrame();
        }

        function drawAnnotatedFrame() {
            const canvas = annotatedCanvas.value;
            const video = reviewVideo.value;
            if (!canvas || !video || video.readyState < 2) return;

            const vw = video.videoWidth;
            const vh = video.videoHeight;
            if (!vw || !vh) return;

            const displayWidth = canvas.clientWidth;
            const displayHeight = displayWidth * (vh / vw);
            canvas.width = displayWidth;
            canvas.height = displayHeight;

            const ctx = canvas.getContext('2d');
            // 左侧原视频镜像绘制
            ctx.drawImage(video, 0, 0, displayWidth, displayHeight);

            // 查找当前时刻对应的检测数据
            const currentTime = video.currentTime;
            const fds = frameDetections.value;
            let bestDetections = [];
            if (fds.length > 0) {
                let closest = fds[0];
                let minDiff = Math.abs(closest.time_sec - currentTime);
                for (let i = 1; i < fds.length; i++) {
                    const diff = Math.abs(fds[i].time_sec - currentTime);
                    if (diff < minDiff) { minDiff = diff; closest = fds[i]; }
                }
                if (minDiff < 1.0) {
                    bestDetections = closest.detections || [];
                }
            }

            const sx = displayWidth / vw;
            const sy = displayHeight / vh;

            // 颜色映射
            const colors = {
                '荷兰猪': '#00FF00', '鹦鹉': '#FF4444', '仓鼠': '#4488FF',
                '奶黄加州': '#00FF80', '三花美泰': '#80FF00', '白加州': '#C8FFC8',
                '食盆': '#FFB400', '水壶': '#0096FF'
            };

            for (const det of bestDetections) {
                const [x1, y1, x2, y2] = det.bbox;
                const bx = x1 * sx, by = y1 * sy, bw = (x2 - x1) * sx, bh = (y2 - y1) * sy;
                const color = colors[det.class_name] || '#FFFFFF';

                ctx.strokeStyle = color;
                ctx.lineWidth = 2;
                ctx.strokeRect(bx, by, bw, bh);

                // 标签
                let label;
                if (det.breed_info && det.breed_info.full_name !== '未知') {
                    label = det.breed_info.color + det.breed_info.breed + ' (' + (det.confidence * 100).toFixed(0) + '%)';
                } else {
                    label = det.class_name + ' ' + det.confidence.toFixed(2);
                }

                ctx.font = 'bold 14px "Microsoft YaHei", "PingFang SC", sans-serif';
                const tm = ctx.measureText(label);
                const labelH = 20, padX = 6;
                const labelY = Math.max(by - labelH - 2, 0);

                // 标签背景
                ctx.fillStyle = color + 'CC';
                ctx.fillRect(bx, labelY, tm.width + padX * 2, labelH);
                // 标签文字
                const rgbSum = parseInt(color.slice(1,3), 16) + parseInt(color.slice(3,5), 16) + parseInt(color.slice(5,7), 16);
                ctx.fillStyle = rgbSum > 380 ? '#000' : '#FFF';
                ctx.fillText(label, bx + padX, labelY + 15);
            }

            // ROI 框
            for (const z of roiZones.value) {
                const zColor = z.name === 'water_bottle' ? '#FFA500' : '#00C865';
                ctx.strokeStyle = zColor;
                ctx.lineWidth = 2;
                ctx.setLineDash([]);
                ctx.strokeRect(z.x, z.y, z.width, z.height);
                ctx.fillStyle = z.name === 'water_bottle' ? 'rgba(255,165,0,0.15)' : 'rgba(0,200,100,0.15)';
                ctx.fillRect(z.x, z.y, z.width, z.height);
                const zLabel = z.name === 'water_bottle' ? '水壶' : '食盆';
                ctx.font = '12px "Microsoft YaHei", sans-serif';
                ctx.fillStyle = zColor;
                ctx.fillText(zLabel, z.x, Math.max(z.y - 5, 15));
            }
        }

        function onVideoLoaded() {
            videoCurrentTime.value = 0;
            videoProgress.value = 0;
            videoPlaying.value = false;
        }

        function onVideoTimeUpdate() {
            const v = reviewVideo.value;
            if (!v || !videoResult.value) return;
            videoCurrentTime.value = v.currentTime;
            videoProgress.value = (v.currentTime / videoResult.value.video_duration_sec) * 100;
        }

        function seekVideo(e) {
            const bar = e.currentTarget;
            const rect = bar.getBoundingClientRect();
            const pct = (e.clientX - rect.left) / rect.width;
            const v = reviewVideo.value;
            if (v && videoResult.value) {
                v.currentTime = pct * videoResult.value.video_duration_sec;
                // Canvas 会在下一帧自动更新到新位置
                drawAnnotatedFrame();
            }
        }

        // ROI 框选
        const roiMode = ref(null);  // 'water_bottle' | 'food_bowl' | null
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
                formData.append("model", selectedModel.value);
                formData.append("conf_threshold", confThreshold.value);

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

        function downloadResult() {
            if (!detectResult.value || !detectResult.value.result_image_url) return;
            const a = document.createElement('a');
            a.href = detectResult.value.result_image_url;
            a.download = 'detection_result.jpg';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
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
                videoResult.value = null;
                monitorActive.value = true;
                healthResult.value = { score: 100, status: '监测中', color: 'success', summary: '等待数据积累', suggestions: [], alerts: [], details: [] };
                await nextTick();
                videoPlayer.value.srcObject = webcamStream;
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
            if (videoPlayer.value) videoPlayer.value.srcObject = null;
            drawEmptyCanvas();
        }

        function drawEmptyCanvas() {
            const canvas = monitorCanvas.value;
            if (!canvas) return;
            canvas.width = canvas.clientWidth;
            canvas.height = canvas.clientWidth * 9 / 16;
            const ctx = canvas.getContext("2d");
            ctx.fillStyle = "#1a1a2e";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            drawROIOnCanvas(ctx);
        }

        function getVideoScale() {
            const c = monitorCanvas.value;
            const v = videoPlayer.value;
            if (!c || !v || !v.videoWidth) return { sx: 1, sy: 1 };
            const ch = c.clientHeight || c.clientWidth * 9 / 16 || 360;
            return { sx: v.videoWidth / c.clientWidth, sy: v.videoHeight / ch };
        }

        function processWebcamFrames() {
            const offCanvas = document.createElement("canvas");
            const video = videoPlayer.value;
            monitorInterval = setInterval(() => {
                if (!monitorActive.value || !wsConnection || wsConnection.readyState !== WebSocket.OPEN) return;
                offCanvas.width = video.videoWidth || 640;
                offCanvas.height = video.videoHeight || 480;
                const ctx = offCanvas.getContext("2d");
                ctx.drawImage(video, 0, 0);
                const frameBase64 = offCanvas.toDataURL("image/jpeg", 0.6);

                // 将 ROI 坐标从画布空间缩放到视频原始分辨率
                const { sx, sy } = getVideoScale();
                const scaledROI = roiZones.value.map(z => ({
                    ...z,
                    x: Math.round(z.x * sx),
                    y: Math.round(z.y * sy),
                    width: Math.round(z.width * sx),
                    height: Math.round(z.height * sy),
                    _orig_x: z.x, _orig_y: z.y, _orig_width: z.width, _orig_height: z.height
                }));

                wsConnection.send(JSON.stringify({
                    frame: frameBase64,
                    roi_zones: scaledROI.map(z => ({ name: z.name, x: z.x, y: z.y, width: z.width, height: z.height })),
                    model: monitorModelKey.value
                }));

                // Canvas 由 WebSocket 回调更新标注图，不在此处绘制
            }, 500);
        }

        async function connectWebSocket() {
            const protocol = location.protocol === "https:" ? "wss:" : "ws:";
            const wsUrl = `${protocol}//${location.host}/api/ws/monitor`;
            wsConnection = new WebSocket(wsUrl);

            wsConnection.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.success && data.result_image_base64) {
                    const monCanvas = monitorCanvas.value;
                    if (!monCanvas) return;
                    const img = new Image();
                    img.onload = () => {
                        const cw = monCanvas.clientWidth;
                        const ch = cw * (img.naturalHeight / img.naturalWidth) || 360;
                        monCanvas.width = cw;
                        monCanvas.height = ch;
                        const ctx = monCanvas.getContext("2d");
                        ctx.drawImage(img, 0, 0, cw, ch);
                        drawROIOnCanvas(ctx);
                    };
                    img.src = "data:image/jpeg;base64," + data.result_image_base64;
                }
                if (data.behavior) {
                    Object.assign(behaviorStats, data.behavior.stats_snapshot || {});
                    // 实时健康评估
                    if (data.behavior.health_alerts && data.behavior.health_alerts.length) {
                        const alerts = data.behavior.health_alerts.map(a => ({
                            severity: a.severity || 'medium',
                            message: a.message,
                            suggestion: a.suggestion || ''
                        }));
                        if (!healthResult.value) {
                            healthResult.value = { score: 70, status: '需关注', color: 'warning', summary: '检测到异常', suggestions: [], alerts, details: [] };
                        } else {
                            healthResult.value.alerts = alerts;
                            if (alerts.some(a => a.severity === 'high')) {
                                healthResult.value.score = Math.min(healthResult.value.score, 70);
                                healthResult.value.status = '需关注';
                                healthResult.value.color = 'warning';
                            }
                        }
                    }
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
                const label = z.name === "water_bottle" ? "水壶" : "食盆";
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
                formData.append("model", monitorModelKey.value);

                const resp = await fetch("/api/video/analyze", { method: "POST", body: formData });
                const data = await resp.json();

                if (data.success) {
                    videoResult.value = data;
                    videoTimeline.value = data.timeline || [];
                    frameDetections.value = data.frame_detections || [];
                    videoProgress.value = 0;
                    videoCurrentTime.value = 0;
                    videoPlaying.value = false;
                    stopCanvasLoop();
                    Object.assign(behaviorStats, data.stats || {});
                    healthResult.value = data.health;
                    updateChart(data.timeline || []);
                    // 首帧绘制
                    nextTick(() => { drawAnnotatedFrame(); });
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
            loadModels();
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
            selectedSpecies, selectedColor, selectedBreed, confThreshold,
            selectedModel, availableModels, breedModels, currentModelInfo,
            speciesList, breedOptions,
            monitorSpecies,
            monitorModelKey, behaviorAvailable,
            topDetection, detectionDisplayName, showBreedCard,
            getSpeciesIcon, handleFileSelect, handleDrop, detectBreed, downloadResult,
            loadModels, onModelChange, modelLabel, modelBadgeClass, modelBorderColor, modelAlertClass, modelDetectionLabel,

            // 健康监护
            videoPlayer, reviewVideo, annotatedCanvas, videoInput, monitorCanvas,
            monitorActive, videoUploading, videoResult,
            videoProgress, videoCurrentTime, videoTimeline, videoPlaying,
            frameDetections,
            formatTime, toggleVideoPlay, onVideoLoaded, onVideoTimeUpdate, seekVideo,
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
