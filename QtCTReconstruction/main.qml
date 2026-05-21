import QtQuick 2.9
import QtQuick.Controls 2.2
import QtQuick.Layouts 1.3
import QtQuick.Window 2.2
import "."

Window {
    id: root
    visible: true
    width: 1400
    height: 950
    title: "CT Reconstruction Viewer"
    color: "#1e1e1e"

    property real updateTicker: 0
    property var pointCloudWindow: null
    property int popupImageIndex: -1

    Connections {
        target: controller
        function onSliceUpdated(index) {
            if (index === controller.currentZ) {
                root.updateTicker = Math.random();
            }
        }
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 10
        spacing: 10

        // Title row
        RowLayout {
            Layout.fillWidth: true
            Layout.preferredHeight: 50

            Label {
                text: "CT Reconstruction Viewer"
                font.pixelSize: 24
                font.bold: true
                color: "white"
            }

            Item { Layout.fillWidth: true }

            Button {
                text: "Generate Volume"
                enabled: !controller.running && !controller.hasVolume
                onClicked: controller.generateVolume()
                implicitWidth: 150
                implicitHeight: 40
                font.pixelSize: 14
            }

            Button {
                text: controller.ready ? "Restart" : "Start Reconstruction"
                enabled: !controller.running && (controller.hasVolume || controller.ready)
                onClicked: controller.startReconstruction()
                implicitWidth: 180
                implicitHeight: 40
                font.pixelSize: 14
            }

            Label {
                text: {
                    if (controller.running) return "Processing...";
                    if (controller.ready) return "Ready";
                    if (controller.hasVolume) return "Volume ready";
                    return "Press Generate";
                }
                font.pixelSize: 14
                color: controller.ready ? "#4caf50" : (controller.running ? "#ff9800" : (controller.hasVolume ? "#2196f3" : "#9e9e9e"))
            }
        }

        // Configuration Row
        RowLayout {
            Layout.fillWidth: true
            Layout.preferredHeight: 40
            spacing: 15

            Label { text: "Backend:"; color: "white" }
            ComboBox {
                id: backendCombo
                // Serial (index 0) — однопоточный эталон для регрессионных тестов,
                // доступен только в Debug-сборке (как кнопка "3D Cloud Viewer").
                // Значения model[] соответствуют BackendType enum:
                //   Serial=0, OpenMP=1, CUDA=2, Hybrid=3
                model: controller.isDebugBuild
                    ? ["Serial (1-thread)", "OpenMP (CPU)", "CUDA (Pure GPU)", "Hybrid (CUDA + OpenMP)"]
                    : ["OpenMP (CPU)", "CUDA (Pure GPU)", "Hybrid (CUDA + OpenMP)"]

                // При использовании debug-списка индекс UI == значение enum.
                // В release-списке Serial нет, поэтому офсет = 1 (Serial пропускаем).
                readonly property int enumOffset: controller.isDebugBuild ? 0 : 1

                currentIndex: controller.backendType - enumOffset

                onCurrentIndexChanged: {
                    const enumVal = currentIndex + enumOffset;
                    if (controller.backendType !== enumVal) {
                        controller.backendType = enumVal;
                    }
                }
                implicitWidth: 185
            }

            Label { text: "Filter:"; color: "white" }
            ComboBox {
                model: ["Ramp", "Shepp-Logan", "Hamming", "Cosine", "Hann", "Bartlett"]
                currentIndex: controller.filterType
                onCurrentIndexChanged: controller.filterType = currentIndex
                implicitWidth: 150
            }

            CheckBox {
                text: "Buffered (Debug)"
                checked: controller.asBuffer
                onCheckedChanged: controller.asBuffer = checked
                // palette.buttonText работает во всех стилях, включая Fusion,
                // в отличие от contentItem: Text, который не поддерживается нативным Windows-стилем
                palette.buttonText: "white"
            }

            Label { text: "Size:"; color: "white" }
            ComboBox {
                model: ["128", "256", "512", "1024"]
                currentIndex: {
                    if (controller.volumeSize === 1024) return 3;
                    if (controller.volumeSize === 512) return 2;
                    if (controller.volumeSize === 256) return 1;
                    return 0; // 128
                }
                onCurrentIndexChanged: controller.volumeSize = parseInt(model[currentIndex], 10)
                implicitWidth: 100
            }

            Item { Layout.fillWidth: true }

            Button {
                text: "Load Point Cloud"
                onClicked: controller.loadPointCloud()
                implicitHeight: 30
            }

            Button {
                text: "Export PNG"
                onClicked: controller.savePng(-1)
                implicitHeight: 30
            }

            Button {
                text: "3D Cloud Viewer"
                visible: controller.isDebugBuild
                enabled: controller.hasVolume
                onClicked: {
                    var component = Qt.createComponent("PointCloudWindow.qml");
                    if (component.status === Component.Ready) {
                        // Сохраняем ссылку в root, чтобы Garbage Collector не закрыл окно
                        if (root.pointCloudWindow) {
                            root.pointCloudWindow.destroy();
                        }
                        root.pointCloudWindow = component.createObject(null, {"ctController": controller});
                        root.pointCloudWindow.show();
                    } else if (component.status === Component.Error) {
                        console.error("Error loading PointCloudWindow:", component.errorString());
                    }
                }
                implicitHeight: 30
            }
        }

        // Timing row
        RowLayout {
            Layout.fillWidth: true
            Layout.preferredHeight: 40
            spacing: 20

            Label {
                text: "Timings:"
                font.pixelSize: 14
                font.bold: true
                color: "#ffeb3b"
            }

            Label {
                text: "Generation: " + controller.genTimeSec.toFixed(3) + " s"
                font.pixelSize: 13
                color: "#81c784"
                Layout.preferredWidth: 150
            }

            Label {
                text: "Sinogram: " + controller.sinogramTimeSec.toFixed(3) + " s"
                font.pixelSize: 13
                color: "#64b5f6"
                Layout.preferredWidth: 150
            }

            Label {
                text: "Reconstruction: " + controller.reconTimeSec.toFixed(3) + " s"
                font.pixelSize: 13
                color: "#ffb74d"
                Layout.preferredWidth: 170
            }

            Label {
                text: "Total: " + (controller.genTimeSec + controller.sinogramTimeSec + controller.reconTimeSec).toFixed(3) + " s"
                font.pixelSize: 13
                font.bold: true
                color: "#e57373"
            }

            Item { Layout.fillWidth: true }
        }

        // Main content - 2x2 grid of images
        GridLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            columns: 2
            rows: 2
            columnSpacing: 10
            rowSpacing: 10

            // Original slice
            ImageWithAxes {
                Layout.fillWidth: true
                Layout.fillHeight: true
                title: "Original"
                accentColor: "#64b5f6"
                source: controller.ready ? ("image://ct/original/" + controller.currentZ + "?t=" + root.updateTicker) : ""
                xMin: -controller.volumeSize / 2; xMax: controller.volumeSize / 2; yMin: -controller.volumeSize / 2; yMax: controller.volumeSize / 2
                xLabel: "X [px]"; yLabel: "Y [px]"
                onClicked: { popupWindowComponent.createObject(root, { "popupImageIndex": 0 }); }
            }

            // Sinogram
            ImageWithAxes {
                Layout.fillWidth: true
                Layout.fillHeight: true
                title: "Sinogram"
                accentColor: "#81c784"
                source: controller.ready ? ("image://ct/sinogram/" + controller.currentZ + "?t=" + root.updateTicker) : ""
                xMin: 0; xMax: 180; yMin: -controller.volumeSize / 2; yMax: controller.volumeSize / 2
                xLabel: "Angle [deg]"; yLabel: "Detector Pos [px]"
                stretchX: true
                onClicked: { popupWindowComponent.createObject(root, { "popupImageIndex": 1 }); }
            }

            // Reconstruction
            ImageWithAxes {
                Layout.fillWidth: true
                Layout.fillHeight: true
                title: "Reconstruction"
                accentColor: "#ffb74d"
                source: controller.ready ? ("image://ct/reconstruction/" + controller.currentZ + "?t=" + root.updateTicker) : ""
                xMin: -controller.volumeSize / 2; xMax: controller.volumeSize / 2; yMin: -controller.volumeSize / 2; yMax: controller.volumeSize / 2
                xLabel: "X [px]"; yLabel: "Y [px]"
                onClicked: { popupWindowComponent.createObject(root, { "popupImageIndex": 2 }); }
            }

            // Difference
            ImageWithAxes {
                Layout.fillWidth: true
                Layout.fillHeight: true
                title: "Difference"
                accentColor: "#aaaaaa"
                source: controller.ready ? ("image://ct/difference/" + controller.currentZ + "?t=" + root.updateTicker) : ""
                xMin: -controller.volumeSize / 2; xMax: controller.volumeSize / 2; yMin: -controller.volumeSize / 2; yMax: controller.volumeSize / 2
                xLabel: "X [px]"; yLabel: "Y [px]"
                showColorScale: true
                colorScaleMaxText: controller.maxDifference.toFixed(1)
                onClicked: { popupWindowComponent.createObject(root, { "popupImageIndex": 3 }); }
            }
        }

        // Slider row
        RowLayout {
            Layout.fillWidth: true
            Layout.preferredHeight: 40
            spacing: 15

            Label {
                text: "Z Slice:"
                font.pixelSize: 14
                color: "white"
                Layout.preferredWidth: 70
            }

            Slider {
                id: zSlider
                Layout.fillWidth: true
                from: 0
                to: controller.maxZ
                value: controller.currentZ
                stepSize: 1
                enabled: controller.ready

                onMoved: {
                    controller.currentZ = Math.round(value)
                }
            }

            Label {
                text: controller.currentZ + " / " + controller.maxZ
                font.pixelSize: 14
                color: "white"
                Layout.preferredWidth: 80
                horizontalAlignment: Text.AlignRight
            }
        }

        // Hounsfield Scale Info
        ColumnLayout {
            Layout.fillWidth: true
            Layout.preferredHeight: 60
            spacing: 5
            
            Label { text: "Hounsfield Scale Reference"; color: "white"; font.bold: true; font.pixelSize: 14 }
            
            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: 15
                gradient: Gradient {
                    orientation: Gradient.Horizontal
                    GradientStop { position: 0.0; color: "black" }
                    GradientStop { position: 1.0; color: "white" }
                }
                border.color: "#555"
                border.width: 1
            }
            
            RowLayout {
                Layout.fillWidth: true
                Label { text: "-1000 (Air)"; color: "#ccc"; font.pixelSize: 12; Layout.alignment: Qt.AlignLeft }
                Item { Layout.fillWidth: true }
                Label { text: "-100...-50 (Fat)"; color: "#ccc"; font.pixelSize: 12; Layout.alignment: Qt.AlignHCenter }
                Item { Layout.fillWidth: true }
                Label { text: "0 (Water)"; color: "#ccc"; font.pixelSize: 12; Layout.alignment: Qt.AlignHCenter }
                Item { Layout.fillWidth: true }
                Label { text: "+40...80 (Blood/Muscle)"; color: "#ccc"; font.pixelSize: 12; Layout.alignment: Qt.AlignHCenter }
                Item { Layout.fillWidth: true }
                Label { text: "+400...+1000 (Bone)"; color: "#ccc"; font.pixelSize: 12; Layout.alignment: Qt.AlignRight }
            }
        }
    }

    Component {
        id: popupWindowComponent
        Window {
            id: instancedWindow
            visible: true
            width: 800
            height: 800
            color: "#1e1e1e"
            
            // This property maps to the component that requested the window
            property int popupImageIndex: 0
            
            title: (popupImage.title !== "" ? popupImage.title : "Detailed View") + " - Slice " + controller.currentZ

            ImageWithAxes {
                id: popupImage
                anchors.fill: parent
                anchors.margins: 15
                
                title: {
                    if (instancedWindow.popupImageIndex === 0) return "Original";
                    if (instancedWindow.popupImageIndex === 1) return "Sinogram";
                    if (instancedWindow.popupImageIndex === 2) return "Reconstruction";
                    if (instancedWindow.popupImageIndex === 3) return "Difference";
                    return "";
                }
                accentColor: {
                    if (instancedWindow.popupImageIndex === 0) return "#64b5f6";
                    if (instancedWindow.popupImageIndex === 1) return "#81c784";
                    if (instancedWindow.popupImageIndex === 2) return "#ffb74d";
                    if (instancedWindow.popupImageIndex === 3) return "#aaaaaa";
                    return "#fff";
                }
                source: {
                    if (!controller.ready || instancedWindow.popupImageIndex < 0) return "";
                    if (instancedWindow.popupImageIndex === 0) return "image://ct/original/" + controller.currentZ + "?t=" + root.updateTicker;
                    if (instancedWindow.popupImageIndex === 1) return "image://ct/sinogram/" + controller.currentZ + "?t=" + root.updateTicker;
                    if (instancedWindow.popupImageIndex === 2) return "image://ct/reconstruction/" + controller.currentZ + "?t=" + root.updateTicker;
                    if (instancedWindow.popupImageIndex === 3) return "image://ct/difference/" + controller.currentZ + "?t=" + root.updateTicker;
                    return "";
                }
                
                xMin: (instancedWindow.popupImageIndex === 1) ? 0 : -controller.volumeSize / 2
                xMax: (instancedWindow.popupImageIndex === 1) ? 180 : controller.volumeSize / 2
                yMin: -controller.volumeSize / 2
                yMax: controller.volumeSize / 2
                

                xLabel: (instancedWindow.popupImageIndex === 1) ? "Angle [deg]" : "X [px]"
                yLabel: (instancedWindow.popupImageIndex === 1) ? "Detector Pos [px]" : "Y [px]"
                
                stretchX: (instancedWindow.popupImageIndex === 1)
                showColorScale: (instancedWindow.popupImageIndex === 3)
                colorScaleMaxText: controller.maxDifference.toFixed(1)
            }
        }
    }
}

