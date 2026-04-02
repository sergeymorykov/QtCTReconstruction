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
    id: window
    width: 800
    height: 600
    visible: true
    title: "3D Point Cloud Debug Viewer"

    property var ctController

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
                farPlane: 1000.0
                position: Qt.vector3d(0.0, 0.0, 250.0)
                upVector: Qt.vector3d(0.0, 1.0, 0.0)
                viewCenter: Qt.vector3d(0.0, 0.0, 0.0)
            }

            OrbitCameraController {
                camera: camera
                lookSpeed: 180
                linearSpeed: 50
            }

            components: [
                RenderSettings {
                    activeFrameGraph: ForwardRenderer {
                        id: renderer
                        clearColor: Qt.rgba(0.1, 0.1, 0.1, 1.0)
                        camera: camera
                    }
                },
                InputSettings { }
            ]

            Entity {
                PointCloudGeometry {
                    id: cloudGeometry
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

    Timer {
        id: initTimer
        interval: 100
        repeat: false
        onTriggered: {
            if (window.ctController) {
                try {
                    window.ctController.extractAndFillPointCloud(cloudGeometry)
                    console.log("3D Point Cloud initialized successfully")
                } catch (e) {
                    console.error("Failed to initialize 3D scene:", e)
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

    Label {
        id: errorLabel
        anchors.centerIn: parent
        text: "OpenGL Error: 3D Visualization failed.\nTry switching to Software/OpenGL rendering."
        color: "red"
        font.pixelSize: 18
        visible: false
        horizontalAlignment: Text.AlignHCenter
    }
}
