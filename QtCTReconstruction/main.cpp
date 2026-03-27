#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>

#include "CtReconstructionController.h"
#include "CtSliceImageProvider.h"

int main(int argc, char *argv[])
{
#if defined(Q_OS_WIN) && QT_VERSION_CHECK(5, 6, 0) <= QT_VERSION && QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
#endif

    QGuiApplication app(argc, argv);

    // Create the reconstruction controller
    CtReconstructionController controller;

    QQmlApplicationEngine engine;

    // Add image provider for CT slice images
    CtSliceImageProvider* imageProvider = new CtSliceImageProvider(&controller);
    engine.addImageProvider(QStringLiteral("ct"), imageProvider);

    // Expose controller to QML context
    engine.rootContext()->setContextProperty(QStringLiteral("controller"), &controller);

    engine.load(QUrl(QStringLiteral("qrc:/qt/qml/qtctreconstruction/main.qml")));
    if (engine.rootObjects().isEmpty())
        return -1;

    return app.exec();
}
