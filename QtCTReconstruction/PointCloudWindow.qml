import QtQuick 2.15
import QtQuick.Window 2.15
import QtQuick.Controls 2.15
import QtQuick.Scene3D 2.15
import Qt3D.Core 2.15
import Qt3D.Render 2.15
import Qt3D.Input 2.15
import Qt3D.Extras 2.15
import QtCTReconstruction 1.0

Window {
    id: rootWindow
    width: 800
    height: 600
    visible: true
    title: "3D Point Cloud Debug Viewer"

    property var ctController

    // Объект, хранящий состояние камеры
    QtObject {
        id: cameraController
        property vector3d orbitCenter: Qt.vector3d(0, 0, 0)
        property real orbitRadius: 100.0
        property real orbitAzimuth: 0.0
        property real orbitElevation: 30.0
        property bool cameraReady: false

        // Флаг для throttle обновлений
        property bool updatePending: false

        function updateCameraPosition() {
            if (!cameraReady || updatePending) return
            updatePending = true
            // Откладываем обновление на следующий кадр, чтобы избежать слишком частых вызовов
            Qt.callLater(doUpdate)
        }

        function doUpdate() {
            updatePending = false
            if (!cameraReady) return
            try {
                var azRad = orbitAzimuth * Math.PI / 180.0
                var elRad = orbitElevation * Math.PI / 180.0
                var x = orbitRadius * Math.cos(elRad) * Math.sin(azRad)
                var y = orbitRadius * Math.sin(elRad)
                var z = orbitRadius * Math.cos(elRad) * Math.cos(azRad)
                if (camera && camera.position) {
                    camera.position = Qt.vector3d(orbitCenter.x + x, orbitCenter.y + y, orbitCenter.z + z)
                    camera.viewCenter = orbitCenter
                    camera.upVector = Qt.vector3d(0, 1, 0)
                }
            } catch(e) {
                console.warn("Camera update error:", e)
            }
        }

        function reset(center, radius) {
            orbitCenter = center
            orbitRadius = radius * 2.8
            orbitAzimuth = 30
            orbitElevation = 40
            updateCameraPosition()
        }
    }

    Scene3D {
        id: scene3d
        anchors.fill: parent
        aspects: ["render", "logic", "input"]
        cameraAspectRatioMode: Scene3D.AutomaticAspectRatio
        focus: true

        Entity {
            id: sceneRoot

            Camera {
                id: camera
                projectionType: CameraLens.PerspectiveProjection
                fieldOfView: 45
                nearPlane: 0.1
                farPlane: 10000.0
                position: Qt.vector3d(0, 0, 100)
                viewCenter: Qt.vector3d(0, 0, 0)
                upVector: Qt.vector3d(0, 1, 0)
            }

            components: [
                RenderSettings {
                    activeFrameGraph: ForwardRenderer {
                        clearColor: Qt.rgba(0.1, 0.1, 0.1, 1.0)
                        camera: camera
                    }
                },
                InputSettings { }
            ]

            Entity {
                id: pointCloudEntity

                PointCloudGeometry {
                    id: cloudGeometry
                    property bool initialResetDone: false

                    onBoundsChanged: {
                        if (hasData && boundsRadius > 0 && !initialResetDone) {
                            console.log("Point cloud loaded. Center:", boundsCenter, "Radius:", boundsRadius)
                            cameraController.cameraReady = true
                            cameraController.reset(boundsCenter, boundsRadius)
                            initialResetDone = true
                        }
                    }
                }

                GeometryRenderer {
                    id: cloudRenderer
                    geometry: cloudGeometry
                    primitiveType: GeometryRenderer.Points
                }

                PerVertexColorMaterial {
                    id: pointMaterial
                }

                components: [cloudRenderer, pointMaterial]
            }
        }
    }

    // MouseArea для управления камерой
    MouseArea {
        id: mouseArea
        anchors.fill: parent
        acceptedButtons: Qt.LeftButton | Qt.RightButton
        property point lastMousePos
        property bool rotating: false

        onPressed: {
            lastMousePos = Qt.point(mouseX, mouseY)
            rotating = true
            cursorShape = Qt.ClosedHandCursor
        }

        onPositionChanged: {
            if (!rotating || !cloudGeometry.hasData || !cameraController.cameraReady) return
            var dx = mouseX - lastMousePos.x
            var dy = mouseY - lastMousePos.y
            cameraController.orbitAzimuth -= dx * 0.5
            cameraController.orbitElevation += dy * 0.5
            if (cameraController.orbitElevation > 89) cameraController.orbitElevation = 89
            if (cameraController.orbitElevation < -89) cameraController.orbitElevation = -89
            cameraController.updateCameraPosition()
            lastMousePos = Qt.point(mouseX, mouseY)
        }

        onReleased: {
            rotating = false
            cursorShape = Qt.ArrowCursor
        }

        onWheel: {
            if (!cloudGeometry.hasData || !cameraController.cameraReady) return
            var delta = wheel.angleDelta.y / 120
            cameraController.orbitRadius -= delta * (cameraController.orbitRadius * 0.1)
            if (cameraController.orbitRadius < 1) cameraController.orbitRadius = 1
            cameraController.updateCameraPosition()
        }
    }

    Timer {
        id: initTimer
        interval: 100
        repeat: false
        onTriggered: {
            if (rootWindow.ctController) {
                try {
                    rootWindow.ctController.extractAndFillPointCloud(cloudGeometry)
                    console.log("Point cloud data sent to geometry")
                } catch (e) {
                    console.error("Failed to initialize point cloud:", e)
                    errorLabel.visible = true
                }
            }
        }
    }

    onVisibleChanged: {
        if (visible) {
            initTimer.start()
        }
    }

    Button {
        text: "🔄 Reset View"
        anchors.top: parent.top
        anchors.right: parent.right
        anchors.margins: 12
        onClicked: {
            if (!cloudGeometry.hasData || !cameraController.cameraReady) return
            cameraController.reset(cloudGeometry.boundsCenter, cloudGeometry.boundsRadius)
        }
    }

    Label {
        id: errorLabel
        anchors.centerIn: parent
        text: "OpenGL Error: 3D Visualization failed.\nTry switching to Software/OpenGL rendering."
        color: "red"
        font.pixelSize: 18
        visible: false
        horizontalAlignment: Text.AlignHCenter
    }

    // Дополнительная защита: при уничтожении окна чистим ссылки
    Component.onDestruction: {
        cameraController.cameraReady = false
    }
}