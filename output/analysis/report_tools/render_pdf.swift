import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

let input = URL(fileURLWithPath: CommandLine.arguments[1])
let output = URL(fileURLWithPath: CommandLine.arguments[2], isDirectory: true)
try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
guard let document = CGPDFDocument(input as CFURL) else { fatalError("Cannot read PDF") }
for number in 1...document.numberOfPages {
    guard let page = document.page(at: number) else { fatalError("Missing page") }
    let box = page.getBoxRect(.mediaBox)
    let scale = 2.0
    let width = Int(ceil(box.width * scale))
    let height = Int(ceil(box.height * scale))
    guard let context = CGContext(data: nil, width: width, height: height,
        bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { fatalError("No context") }
    context.setFillColor(CGColor(gray: 1, alpha: 1))
    context.fill(CGRect(x: 0, y: 0, width: width, height: height))
    context.scaleBy(x: scale, y: scale)
    context.translateBy(x: -box.origin.x, y: -box.origin.y)
    context.drawPDFPage(page)
    guard let image = context.makeImage() else { fatalError("No image") }
    let path = output.appendingPathComponent(String(format: "page-%02d.png", number))
    guard let dest = CGImageDestinationCreateWithURL(path as CFURL, UTType.png.identifier as CFString, 1, nil) else { fatalError("No destination") }
    CGImageDestinationAddImage(dest, image, nil)
    guard CGImageDestinationFinalize(dest) else { fatalError("Cannot write PNG") }
}
print("Rendered \(document.numberOfPages) pages")
