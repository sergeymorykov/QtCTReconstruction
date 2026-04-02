import QtQuick 2.9
import QtQuick.Controls 2.2
import QtQuick.Layouts 1.3
import QtQuick.Window 2.2

Window {
    id: root
    visible: true
    width: 1400
    height: 950
    title: "CT Reconstruction Viewer"
    color: "#1e1e1e"

    property string imagePrefix: controller.ready ? "image://ct/" : ""
    property real updateTicker: 0

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
                model: ["OpenMP (CPU)", "CUDA (GPU)"]
                currentIndex: controller.backendType
                onCurrentIndexChanged: {
                    console.log("Selected backend:", currentIndex);
                    controller.backendType = currentIndex;
                }
                implicitWidth: 150
            }

            Label { text: "Filter:"; color: "white" }
            ComboBox {
                model: ["Ramp", "Shepp-Logan", "Hamming"]
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
                        // null-родитель обязателен для отдельного Window;
                        // с root-родителем Scene3D не попадает в graphics scene
                        var pointCloudWindow = component.createObject(null, {"ctController": controller});
                        pointCloudWindow.show();
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
            ColumnLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                spacing: 5

                Label {
                    text: "Original"
                    font.pixelSize: 16
                    font.bold: true
                    color: "#64b5f6"
                    Layout.preferredHeight: 25
                }

                Image {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    fillMode: Image.PreserveAspectFit
                    smooth: true
                    source: controller.ready ? (imagePrefix + "original/" + controller.currentZ + "?t=" + root.updateTicker) : ""
                    cache: false

                    Rectangle {
                        anchors.fill: parent
                        color: "transparent"
                        border.width: 2
                        border.color: "#64b5f6"
                        visible: controller.ready
                    }
                }
            }

            // Sinogram
            ColumnLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                spacing: 5

                Label {
                    text: "Sinogram"
                    font.pixelSize: 16
                    font.bold: true
                    color: "#81c784"
                    Layout.preferredHeight: 25
                }

                Image {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    fillMode: Image.PreserveAspectFit
                    smooth: true
                    source: controller.ready ? (imagePrefix + "sinogram/" + controller.currentZ + "?t=" + root.updateTicker) : ""
                    cache: false

                    Rectangle {
                        anchors.fill: parent
                        color: "transparent"
                        border.width: 2
                        border.color: "#81c784"
                        visible: controller.ready
                    }
                }
            }

            // Reconstruction
            ColumnLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                spacing: 5

                Label {
                    text: "Reconstruction"
                    font.pixelSize: 16
                    font.bold: true
                    color: "#ffb74d"
                    Layout.preferredHeight: 25
                }

                Image {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    fillMode: Image.PreserveAspectFit
                    smooth: true
                    source: controller.ready ? (imagePrefix + "reconstruction/" + controller.currentZ + "?t=" + root.updateTicker) : ""
                    cache: false

                    Rectangle {
                        anchors.fill: parent
                        color: "transparent"
                        border.width: 2
                        border.color: "#ffb74d"
                        visible: controller.ready
                    }
                }
            }

            // Difference
            ColumnLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                spacing: 5

                Label {
                    text: "Difference"
                    font.pixelSize: 16
                    font.bold: true
                    color: "#e57373"
                    Layout.preferredHeight: 25
                }

                Image {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    fillMode: Image.PreserveAspectFit
                    smooth: true
                    source: controller.ready ? (imagePrefix + "difference/" + controller.currentZ + "?t=" + root.updateTicker) : ""
                    cache: false

                    Rectangle {
                        anchors.fill: parent
                        color: "transparent"
                        border.width: 2
                        border.color: "#e57373"
                        visible: controller.ready
                    }
                }
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
}
