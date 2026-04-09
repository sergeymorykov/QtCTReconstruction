import QtQuick 2.9
import QtQuick.Controls 2.2
import QtQuick.Layouts 1.3

Item {
    id: root
    property string title: ""
    property var source: ""
    property real xMin: -128
    property real xMax: 128
    property real yMin: -128
    property real yMax: 128
    property string xLabel: "X"
    property string yLabel: "Y"
    property bool stretchX: false
    property bool showColorScale: false
    property color accentColor: "#64b5f6"

    signal clicked()

    implicitWidth: 400
    implicitHeight: 400

    MouseArea {
        anchors.fill: parent
        cursorShape: Qt.PointingHandCursor
        onClicked: root.clicked()
        // Prevent click events from propagating to underlying items if necessary, but keep it basic
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 5

        // Header Title
        Label {
            text: root.title
            font.pixelSize: 16
            font.bold: true
            color: root.accentColor
            Layout.alignment: Qt.AlignLeft
        }

        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 10

            // Y Axis Label (Vertical)
            Item {
                Layout.preferredWidth: 20
                Layout.fillHeight: true
                Label {
                    text: root.yLabel
                    color: "#aaa"
                    font.pixelSize: 12
                    rotation: -90
                    anchors.centerIn: parent
                }
            }

            // Y Axis Ticks and Values
            ColumnLayout {
                id: yAxisItems
                Layout.preferredWidth: 35
                Layout.fillHeight: true
                spacing: 0
                
                // Top value
                Label { text: root.yMax.toFixed(0); color: "#888"; font.pixelSize: 10; Layout.alignment: Qt.AlignRight }
                Item { Layout.fillHeight: true }
                // Middle value
                Label { text: ((root.yMax + root.yMin)/2).toFixed(0); color: "#888"; font.pixelSize: 10; Layout.alignment: Qt.AlignRight }
                Item { Layout.fillHeight: true }
                // Bottom value
                Label { text: root.yMin.toFixed(0); color: "#888"; font.pixelSize: 10; Layout.alignment: Qt.AlignRight }
            }

            // Main Image Area
            Rectangle {
                id: imageContainer
                Layout.fillWidth: true
                Layout.fillHeight: true
                color: "#121212"
                border.width: 2
                border.color: root.accentColor
                clip: true

                Image {
                    id: mainImage
                    anchors.centerIn: parent
                    width: root.stretchX ? parent.width - 4 : Math.min(parent.width - 4, (parent.height - 4) * (implicitWidth / (implicitHeight || 1)))
                    height: root.stretchX ? parent.height - 4 : Math.min(parent.height - 4, (parent.width - 4) * (implicitHeight / (implicitWidth || 1)))
                    fillMode: root.stretchX ? Image.Stretch : Image.PreserveAspectFit
                    smooth: true
                    source: root.source
                    cache: false
                }
            }

            // Optional Color Scale
            ColumnLayout {
                visible: root.showColorScale
                Layout.preferredWidth: 60
                Layout.fillHeight: true
                spacing: 2

                Label { text: "50"; color: "#aaa"; font.pixelSize: 10; Layout.alignment: Qt.AlignHCenter }

                Rectangle {
                    Layout.fillHeight: true
                    Layout.preferredWidth: 15
                    Layout.alignment: Qt.AlignHCenter
                    gradient: Gradient {
                        GradientStop { position: 0.0; color: "white" }
                        GradientStop { position: 0.5; color: "gray" }
                        GradientStop { position: 1.0; color: "black" }
                    }
                    border.color: "#444"
                }

                Label { text: "0"; color: "#aaa"; font.pixelSize: 10; Layout.alignment: Qt.AlignHCenter }
                Label { text: "Abs"; color: "#aaa"; font.pixelSize: 10; Layout.alignment: Qt.AlignHCenter }

            }
        }

        // X Axis Area
        RowLayout {
            Layout.fillWidth: true
            Layout.preferredHeight: 40
            spacing: 0

            // Placeholder for Y axis labels
            Item { Layout.preferredWidth: 20 + 35 + 10 }

            ColumnLayout {
                Layout.fillWidth: true
                spacing: 2

                // X Ticks and Labels
                RowLayout {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 15
                    spacing: 0
                    
                    Label { text: root.xMin.toFixed(0); color: "#888"; font.pixelSize: 10; Layout.alignment: Qt.AlignLeft }
                    Item { Layout.fillWidth: true }
                    Label { text: root.xMax.toFixed(0); color: "#888"; font.pixelSize: 10; Layout.alignment: Qt.AlignRight }
                }

                Label {
                    text: root.xLabel
                    font.pixelSize: 12
                    color: "#aaa"
                    Layout.alignment: Qt.AlignHCenter
                }
            }

            // Placeholder for color scale
            Item { Layout.preferredWidth: root.showColorScale ? 60 + 10 : 0 }
        }
    }
}
