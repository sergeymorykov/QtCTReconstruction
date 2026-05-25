import QtQuick 2.9
import QtQuick.Controls 2.2
import QtQuick.Layouts 1.3
import QtQuick.Window 2.2
import "."

// CT Reconstruction Viewer — Consumer build.
// Упрощённый UI: загрузка готовых синограмм с диска → FBP-реконструкция →
// просмотр срезов по Z-слайдеру → экспорт PNG.
// Без Generate/Original/Sinogram/Difference — этих данных у Consumer нет.

Window {
    id: root
    visible: true
    width: 1100
    height: 850
    title: "CT Reconstruction Viewer"
    color: "#1e1e1e"

    property real updateTicker: 0

    // Consumer-режим всегда использует CUDA — выбор backend'а из UI убран.
    // Принудительно ставим CUDA (enum value 2 в BackendType) при старте.
    Component.onCompleted: {
        controller.backendType = 2;
    }

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

        // ------------------------------------------------------------
        // Title row
        // ------------------------------------------------------------
        RowLayout {
            Layout.fillWidth: true
            Layout.preferredHeight: 50

            Label {
                text: "CT Reconstruction Viewer — Consumer"
                font.pixelSize: 24
                font.bold: true
                color: "white"
            }

            Item { Layout.fillWidth: true }

            Label {
                text: {
                    if (controller.running) return "Processing...";
                    if (controller.ready) return "Ready";
                    if (controller.hasSinograms) return "Sinograms loaded";
                    return "Load sinograms (.npy)";
                }
                font.pixelSize: 14
                color: controller.ready ? "#4caf50"
                       : (controller.running ? "#ff9800"
                       : (controller.hasSinograms ? "#2196f3" : "#9e9e9e"))
            }
        }

        // ------------------------------------------------------------
        // Configuration + buttons row
        // ------------------------------------------------------------
        RowLayout {
            Layout.fillWidth: true
            Layout.preferredHeight: 40
            spacing: 15

            // Backend в Consumer-режиме всегда CUDA — UI-контрол убран
            // (см. Component.onCompleted в Window).

            Label { text: "Filter:"; color: "white" }
            ComboBox {
                model: ["Ramp", "Shepp-Logan", "Hamming", "Cosine", "Hann", "Bartlett"]
                currentIndex: controller.filterType
                onCurrentIndexChanged: controller.filterType = currentIndex
                implicitWidth: 150
            }

            Item { Layout.fillWidth: true }

            Button {
                text: "Load Sinograms"
                enabled: !controller.running
                onClicked: {
                    if (controller.loadSinograms()) {
                        // Состояние обновится через сигналы; UI среагирует автоматически.
                    }
                }
                implicitHeight: 30
            }

            Button {
                text: controller.ready ? "Restart" : "Start Reconstruction"
                enabled: !controller.running && controller.hasSinograms
                onClicked: controller.startReconstruction()
                implicitHeight: 30
            }

            Button {
                text: "Export PNG"
                enabled: controller.ready
                onClicked: controller.exportAllReconstructionPng()
                implicitHeight: 30
            }
        }

        // ------------------------------------------------------------
        // Timing row
        // ------------------------------------------------------------
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
                text: "Reconstruction: " + controller.reconTimeSec.toFixed(3) + " s"
                font.pixelSize: 13
                color: "#ffb74d"
            }

            Item { Layout.fillWidth: true }
        }

        // ------------------------------------------------------------
        // Главный image area — квадрат по меньшей стороне, по центру.
        // Item-обёртка занимает всё доступное пространство (fillWidth+fillHeight),
        // а внутренний ImageWithAxes — квадрат min(width, height), центрирован.
        // Так рамка всегда соответствует пропорциям изображения (1:1).
        // ------------------------------------------------------------
        Item {
            Layout.fillWidth: true
            Layout.fillHeight: true

            ImageWithAxes {
                id: reconImage
                anchors.centerIn: parent
                width:  Math.min(parent.width, parent.height)
                height: width
                title: "Reconstruction"
                accentColor: "#ffb74d"
                source: controller.ready
                        ? ("image://ct/reconstruction/" + controller.currentZ + "?t=" + root.updateTicker)
                        : ""
                // Оси берутся из ФАКТИЧЕСКОГО размера загруженного изображения
                // (sourceSize из QML Image). Пока картинка не загружена —
                // sourcePixelWidth/Height = 0; делаем fallback на volumeSize,
                // чтобы стартовая надпись была разумной.
                readonly property int imgW: reconImage.sourcePixelWidth  > 0
                                            ? reconImage.sourcePixelWidth
                                            : controller.volumeSize
                readonly property int imgH: reconImage.sourcePixelHeight > 0
                                            ? reconImage.sourcePixelHeight
                                            : controller.volumeSize
                xMin: -imgW / 2
                xMax:  imgW / 2
                yMin: -imgH / 2
                yMax:  imgH / 2
                xLabel: "X [px]"
                yLabel: "Y [px]"
            }
        }

        // ------------------------------------------------------------
        // Z slider
        // ------------------------------------------------------------
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
                onMoved: controller.currentZ = Math.round(value)
            }

            Label {
                text: controller.currentZ + " / " + controller.maxZ
                font.pixelSize: 14
                color: "white"
                Layout.preferredWidth: 80
                horizontalAlignment: Text.AlignRight
            }
        }
    }
}
