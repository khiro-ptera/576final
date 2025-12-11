import java.awt.*;
import java.awt.geom.*;
import java.awt.image.*;
import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import javax.swing.*;

public class ImageDisplay {
    JFrame frame;
    JLabel lbIm1;
    BufferedImage imgOne;
    int width;
    int height;

    String imagePath;

    // Detected pieces
    List<PuzzlePiece> pieces = new ArrayList<>();

    // Class to represent piece
    static class PuzzlePiece {
        BufferedImage image;   // final, axis-aligned, rectangular tile
        Rectangle bounds;      // original bounding box in source image
        int id;
        double rotationAngle;  // angle (in degrees) applied to make it axis-aligned
        
        // For puzzle solving
        Point initialPosition = new Point(0, 0);   // (x, y) for center of tile
        Point finalPosition = new Point(0, 0);
        int gridRow = 0, gridCol = 0;              // grid position after solving
        double initialRotation = 0;
        double finalRotation = 0;
        int additionalRotation = 0;                // Additional 90° rotations needed (0, 90, 180, 270)

        PuzzlePiece(BufferedImage image, Rectangle bounds, int id) {
            this.image = image;
            this.bounds = bounds;
            this.id = id;
            this.rotationAngle = 0;
        }
    }
    
    // Edge types for matching
    enum EdgeType {
        TOP, BOTTOM, LEFT, RIGHT
    }
    
    // Class to represent an edge match between two pieces
    static class EdgeMatch implements Comparable<EdgeMatch> {
        int piece1;
        int piece2;
        EdgeType edge1;
        EdgeType edge2;
        double score;
        
        EdgeMatch(int piece1, int piece2, EdgeType edge1, EdgeType edge2, double score) {
            this.piece1 = piece1;
            this.piece2 = piece2;
            this.edge1 = edge1;
            this.edge2 = edge2;
            this.score = score;
        }
        
        @Override
        public int compareTo(EdgeMatch other) {
            return Double.compare(other.score, this.score);  // Higher scores first
        }
        
        @Override
        public String toString() {
            return String.format("Piece %d (%s) <-> Piece %d (%s): %.4f",
                    piece1, edge1, piece2, edge2, score);
        }
    }
    
    // Placement edge connecting two pieces
    static class PlacementEdge {
        int otherId;
        EdgeType otherEdge;
        double score;
        
        PlacementEdge(int otherId, EdgeType otherEdge, double score) {
            this.otherId = otherId;
            this.otherEdge = otherEdge;
            this.score = score;
        }
    }
    
    // Grid position for a piece
    static class GridPos {
        int row, col;
        GridPos(int row, int col) {
            this.row = row;
            this.col = col;
        }
        
        @Override
        public boolean equals(Object o) {
            if (!(o instanceof GridPos)) return false;
            GridPos gp = (GridPos) o;
            return row == gp.row && col == gp.col;
        }
        
        @Override
        public int hashCode() {
            return row * 31 + col;
        }
    }

    // ~~~~~~~~~~~~~~~
    // Basic I/O
    public void readImageRGB(int width, int height, String imgPath, BufferedImage img) {
        try (RandomAccessFile raf = new RandomAccessFile(imgPath, "r")) {
            int frameLength = width * height * 3;
            byte[] bytes = new byte[frameLength];
            raf.read(bytes);

            int ind = 0;
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    byte r = bytes[ind];
                    byte g = bytes[ind + height * width];
                    byte b = bytes[ind + height * width * 2];
                    int pix = 0xff000000 | ((r & 0xff) << 16) | ((g & 0xff) << 8) | (b & 0xff);
                    img.setRGB(x, y, pix);
                    ind++;
                }
            }
        } catch (Exception e) {
            System.err.println("Error reading image: " + e.getMessage());
        }
    }

    private void parseArgs(String[] args) {
        imagePath = args[0];
        width = Integer.parseInt(args[1]);
        height = Integer.parseInt(args[2]);
    }

    // ~~~~~~~~~~~~~~~
    // Piece detection

    public void detectPieces() {
        boolean[][] mask = createForegroundMask();

        boolean[][] visited = new boolean[height][width];
        int pieceId = 0;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (mask[y][x] && !visited[y][x]) {
                    PuzzlePiece piece = extractPiece(x, y, mask, visited, pieceId);
                    if (piece != null && piece.bounds.width > 10 && piece.bounds.height > 10) {
                        pieces.add(piece);
                        pieceId++;
                    }
                }
            }
        }

        System.out.println("Detected " + pieces.size() + " puzzle pieces");
        for (PuzzlePiece piece : pieces) {
            System.out.println("Piece " + piece.id + " - Rotation applied: "
                    + piece.rotationAngle + " degrees");
        }
    }

    private boolean[][] createForegroundMask() {
        boolean[][] mask = new boolean[height][width];

        int bgColor = estimateBackgroundColor();
        int threshold = 30;

        int bgR = (bgColor >> 16) & 0xFF;
        int bgG = (bgColor >> 8) & 0xFF;
        int bgB = bgColor & 0xFF;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = imgOne.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;

                int diff = Math.abs(r - bgR) + Math.abs(g - bgG) + Math.abs(b - bgB);
                mask[y][x] = diff > threshold;
            }
        }

        return mask;
    }

    private int estimateBackgroundColor() {
        int[] corners = new int[4];
        corners[0] = imgOne.getRGB(0, 0);
        corners[1] = imgOne.getRGB(width - 1, 0);
        corners[2] = imgOne.getRGB(0, height - 1);
        corners[3] = imgOne.getRGB(width - 1, height - 1);
        return corners[0]; // uniform background in your inputs
    }

    private PuzzlePiece extractPiece(int startX, int startY,
                                     boolean[][] mask,
                                     boolean[][] visited,
                                     int id) {
        Queue<Point> queue = new LinkedList<>();
        List<Point> piecePixels = new ArrayList<>();

        queue.add(new Point(startX, startY));
        visited[startY][startX] = true;

        int minX = startX, maxX = startX;
        int minY = startY, maxY = startY;

        while (!queue.isEmpty()) {
            Point p = queue.poll();
            piecePixels.add(p);

            minX = Math.min(minX, p.x);
            maxX = Math.max(maxX, p.x);
            minY = Math.min(minY, p.y);
            maxY = Math.max(maxY, p.y);

            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    int nx = p.x + dx;
                    int ny = p.y + dy;

                    if (nx >= 0 && nx < width && ny >= 0 && ny < height &&
                            mask[ny][nx] && !visited[ny][nx]) {
                        visited[ny][nx] = true;
                        queue.add(new Point(nx, ny));
                    }
                }
            }
        }

        Rectangle bounds = new Rectangle(minX, minY,
                maxX - minX + 1, maxY - minY + 1);

        List<Point> localPixels = new ArrayList<>();
        for (Point p : piecePixels) {
            int lx = p.x - bounds.x;
            int ly = p.y - bounds.y;
            localPixels.add(new Point(lx, ly));
        }

        double rotationAngle = detectRotationFromContour(piecePixels, bounds);
        BufferedImage pieceImage = extractAndRotatePiece(localPixels, bounds, rotationAngle);

        PuzzlePiece piece = new PuzzlePiece(pieceImage, bounds, id);
        piece.rotationAngle = rotationAngle; // keep for logging
        piece.initialPosition = new Point(bounds.x + bounds.width / 2, bounds.y + bounds.height / 2);
        // Use detected angle so animation shows rotation back to upright
        piece.initialRotation = rotationAngle;
        return piece;
    }

    // Detect rotation
    private double detectRotationFromContour(List<Point> pixels, Rectangle bounds) {
        if (pixels.isEmpty()) return 0.0;
        
        // Find the minimum area rectangle
        int n = pixels.size();
        double[] xs = new double[n];
        double[] ys = new double[n];

        for (int i = 0; i < n; i++) {
            xs[i] = pixels.get(i).x;
            ys[i] = pixels.get(i).y;
        }

        double bestAngle = 0.0;
        double bestArea = Double.MAX_VALUE;
        double bestWidth = 0, bestHeight = 0;

        // Try angles to find minimum area rectangle
        for (double angle = -89.0; angle <= 89.0; angle += 1.0) {
            double rad = Math.toRadians(angle);
            double cos = Math.cos(rad);
            double sin = Math.sin(rad);

            double minX = Double.POSITIVE_INFINITY;
            double maxX = Double.NEGATIVE_INFINITY;
            double minY = Double.POSITIVE_INFINITY;
            double maxY = Double.NEGATIVE_INFINITY;

            for (int i = 0; i < n; i++) {
                double rx = xs[i] * cos - ys[i] * sin;
                double ry = xs[i] * sin + ys[i] * cos;

                if (rx < minX) minX = rx;
                if (rx > maxX) maxX = rx;
                if (ry < minY) minY = ry;
                if (ry > maxY) maxY = ry;
            }

            double w = maxX - minX;
            double h = maxY - minY;
            double area = w * h;

            if (area < bestArea) {
                bestArea = area;
                bestAngle = angle;
                bestWidth = w;
                bestHeight = h;
            }
        }
        
        // compute the 4 corners
        double rad = Math.toRadians(bestAngle);
        double cos = Math.cos(rad);
        double sin = Math.sin(rad);
        
        // Find bounds in rotated coordinate system
        double minX = Double.POSITIVE_INFINITY;
        double maxX = Double.NEGATIVE_INFINITY;
        double minY = Double.POSITIVE_INFINITY;
        double maxY = Double.NEGATIVE_INFINITY;
        
        for (int i = 0; i < n; i++) {
            double rx = xs[i] * cos - ys[i] * sin;
            double ry = xs[i] * sin + ys[i] * cos;
            if (rx < minX) minX = rx;
            if (rx > maxX) maxX = rx;
            if (ry < minY) minY = ry;
            if (ry > maxY) maxY = ry;
        }
        
        // Then transform back to original coords
        Point2D.Double[] corners = new Point2D.Double[4];
        double[][] rotCorners = {
            {minX, minY}, {maxX, minY}, {maxX, maxY}, {minX, maxY}
        };
        
        for (int i = 0; i < 4; i++) {
            double rx = rotCorners[i][0];
            double ry = rotCorners[i][1];
            double ox = rx * cos + ry * sin;  // rotate back
            double oy = -rx * sin + ry * cos;
            corners[i] = new Point2D.Double(ox, oy);
        }
        
        // Sort corners by y-coordinate
        Point2D.Double[] sortedCorners = corners.clone();
        java.util.Arrays.sort(sortedCorners, (c1, c2) -> Double.compare(c1.y, c2.y));
        
        if (sortedCorners[2].x < sortedCorners[0].x) {
            bestAngle += 90;
        }
        
        // Normalize to [-45, 45] range
        while (bestAngle > 45) bestAngle -= 90;
        while (bestAngle < -45) bestAngle += 90;
        
        return bestAngle;
    }

    private BufferedImage extractAndRotatePiece(List<Point> pixels,
                                                Rectangle bounds,
                                                double angleDegrees) {

        // extract local ARGB subimage
        BufferedImage original = new BufferedImage(bounds.width, bounds.height,
                BufferedImage.TYPE_INT_ARGB);

        for (Point p : pixels) {
            int lx = p.x;
            int ly = p.y;
            int sx = bounds.x + lx;
            int sy = bounds.y + ly;
            original.setRGB(lx, ly, imgOne.getRGB(sx, sy));
        }

        // if ~0°, just crop and convert
        if (Math.abs(angleDegrees) < 1e-3) {
            BufferedImage cropped0 = cropToContent(original);
            return argbToRgb(cropped0);
        }

        double angle = Math.toRadians(angleDegrees);

        double sin = Math.abs(Math.sin(angle));
        double cos = Math.abs(Math.cos(angle));
        int newW = (int) Math.ceil(bounds.width * cos + bounds.height * sin);
        int newH = (int) Math.ceil(bounds.height * cos + bounds.width * sin);

        BufferedImage rotated = new BufferedImage(newW, newH, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g2d = rotated.createGraphics();
        g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                RenderingHints.VALUE_INTERPOLATION_BILINEAR);

        g2d.setComposite(AlphaComposite.Clear);
        g2d.fillRect(0, 0, newW, newH);
        g2d.setComposite(AlphaComposite.Src);

        g2d.translate(newW / 2.0, newH / 2.0);
        g2d.rotate(angle);
        g2d.translate(-bounds.width / 2.0, -bounds.height / 2.0);
        g2d.drawImage(original, 0, 0, null);
        g2d.dispose();

        // Remove 1-pixel border from rotation artifacts
        BufferedImage cropped = cropToContent(rotated);
        
        // Clean anti-aliased edges by removing low-alpha pixels at boundaries
        BufferedImage cleaned = cleanBorders(cropped);
        
        BufferedImage rgb = argbToRgb(cleaned);

        return rgb;
    }
    
    /**
     * Clean anti-aliased borders by removing anti-aliasing artifacts.
     * This removes the fuzzy black edges created by bilinear interpolation during rotation.
     */
    private BufferedImage cleanBorders(BufferedImage img) {
        int w = img.getWidth();
        int h = img.getHeight();
        
        // If image is too small to crop, return as-is
        if (w <= 2 || h <= 2) {
            return img;
        }
        
        // Crop exactly 1 pixel from each side
        int newW = w - 2;
        int newH = h - 2;
        
        BufferedImage cleaned = new BufferedImage(newW, newH, BufferedImage.TYPE_INT_ARGB);
        for (int y = 1; y < h - 1; y++) {
            for (int x = 1; x < w - 1; x++) {
                int argb = img.getRGB(x, y);
                cleaned.setRGB(x - 1, y - 1, argb);
            }
        }
        
        return cleaned;
    }

    private BufferedImage cropToContent(BufferedImage img) {
        int w = img.getWidth();
        int h = img.getHeight();

        int minX = w, minY = h, maxX = 0, maxY = 0;
        boolean found = false;

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int a = (img.getRGB(x, y) >> 24) & 0xFF;
                if (a > 10) {
                    found = true;
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                }
            }
        }

        if (!found) return img;
        return img.getSubimage(minX, minY, maxX - minX + 1, maxY - minY + 1);
    }

    private BufferedImage argbToRgb(BufferedImage src) {
        int w = src.getWidth();
        int h = src.getHeight();
        BufferedImage dst = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int argb = src.getRGB(x, y);
                int a = (argb >> 24) & 0xFF;
                if (a < 10) {
                    dst.setRGB(x, y, 0x000000);
                } else {
                    dst.setRGB(x, y, argb & 0x00FFFFFF);
                }
            }
        }
        return dst;
    }

    /**
     * Add a small replicated border around the image to help fill tiny gaps
     * between tiles when composing the final puzzle. Pads by `pad` pixels on
     * all sides by copying edge pixels outward.
     */
    private BufferedImage replicateBorderPixels(BufferedImage src, int pad) {
        if (pad <= 0) return src;
        int w = src.getWidth();
        int h = src.getHeight();
        BufferedImage out = new BufferedImage(w + 2 * pad, h + 2 * pad, src.getType());

        // Fill center
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                out.setRGB(x + pad, y + pad, src.getRGB(x, y));
            }
        }

        // Helper to decide if a pixel is background-ish
        int bgRGB = estimateBackgroundColor();
        java.util.function.IntPredicate isBg = (rgb) -> {
            int r = (rgb >> 16) & 0xFF;
            int g = (rgb >> 8) & 0xFF;
            int b = rgb & 0xFF;
            // near-black from prior alpha cleanup
            boolean nearBlack = r < 8 && g < 8 && b < 8;
            // close to global background
            int br = (bgRGB >> 16) & 0xFF;
            int bgc = (bgRGB >> 8) & 0xFF;
            int bb = bgRGB & 0xFF;
            int diff = Math.abs(r - br) + Math.abs(g - bgc) + Math.abs(b - bb);
            return nearBlack || diff < 40;
        };

        // Top and bottom rows (replicate)
        for (int x = 0; x < w; x++) {
            int topPix = src.getRGB(x, 0);
            int botPix = src.getRGB(x, h - 1);
            // Replace background-looking pixels with inner neighbor
            if (isBg.test(topPix) && h > 1) topPix = src.getRGB(x, Math.min(1, h - 1));
            if (isBg.test(botPix) && h > 1) botPix = src.getRGB(x, Math.max(h - 2, 0));
            for (int p = 0; p < pad; p++) {
                out.setRGB(x + pad, p, topPix);
                out.setRGB(x + pad, h + pad + p, botPix);
            }
        }

        // Left and right columns (replicate)
        for (int y = 0; y < h; y++) {
            int leftPix = src.getRGB(0, y);
            int rightPix = src.getRGB(w - 1, y);
            if (isBg.test(leftPix) && w > 1) leftPix = src.getRGB(Math.min(1, w - 1), y);
            if (isBg.test(rightPix) && w > 1) rightPix = src.getRGB(Math.max(w - 2, 0), y);
            for (int p = 0; p < pad; p++) {
                out.setRGB(p, y + pad, leftPix);
                out.setRGB(w + pad + p, y + pad, rightPix);
            }
        }

        // use inner neighbor if corner looks like background
        int tl = src.getRGB(0, 0);
        int tr = src.getRGB(w - 1, 0);
        int bl = src.getRGB(0, h - 1);
        int br = src.getRGB(w - 1, h - 1);
        if (isBg.test(tl) && w > 1 && h > 1) tl = src.getRGB(Math.min(1, w - 1), Math.min(1, h - 1));
        if (isBg.test(tr) && w > 1 && h > 1) tr = src.getRGB(Math.max(w - 2, 0), Math.min(1, h - 1));
        if (isBg.test(bl) && w > 1 && h > 1) bl = src.getRGB(Math.min(1, w - 1), Math.max(h - 2, 0));
        if (isBg.test(br) && w > 1 && h > 1) br = src.getRGB(Math.max(w - 2, 0), Math.max(h - 2, 0));
        for (int pY = 0; pY < pad; pY++) {
            for (int pX = 0; pX < pad; pX++) {
                out.setRGB(pX, pY, tl);
                out.setRGB(w + pad + pX, pY, tr);
                out.setRGB(pX, h + pad + pY, bl);
                out.setRGB(w + pad + pX, h + pad + pY, br);
            }
        }

        return out;
    }

    // ~~~~~~~~~~~~~~~
    // Simple MSE-based Edge Matching 
    
    private int[] extractEdgePixels(BufferedImage img, EdgeType edge) {
        int h = img.getHeight();
        int w = img.getWidth();
        int[] pixels = null;
        
        switch (edge) {
            case TOP:
                pixels = new int[w * 3];  // RGB channels
                for (int x = 0; x < w; x++) {
                    int rgb = img.getRGB(x, 0);
                    pixels[x * 3] = (rgb >> 16) & 0xFF;     // R
                    pixels[x * 3 + 1] = (rgb >> 8) & 0xFF;  // G
                    pixels[x * 3 + 2] = rgb & 0xFF;         // B
                }
                break;
            case BOTTOM:
                pixels = new int[w * 3];
                for (int x = 0; x < w; x++) {
                    int rgb = img.getRGB(x, h - 1);
                    pixels[x * 3] = (rgb >> 16) & 0xFF;
                    pixels[x * 3 + 1] = (rgb >> 8) & 0xFF;
                    pixels[x * 3 + 2] = rgb & 0xFF;
                }
                break;
            case LEFT:
                pixels = new int[h * 3];
                for (int y = 0; y < h; y++) {
                    int rgb = img.getRGB(0, y);
                    pixels[y * 3] = (rgb >> 16) & 0xFF;
                    pixels[y * 3 + 1] = (rgb >> 8) & 0xFF;
                    pixels[y * 3 + 2] = rgb & 0xFF;
                }
                break;
            case RIGHT:
                pixels = new int[h * 3];
                for (int y = 0; y < h; y++) {
                    int rgb = img.getRGB(w - 1, y);
                    pixels[y * 3] = (rgb >> 16) & 0xFF;
                    pixels[y * 3 + 1] = (rgb >> 8) & 0xFF;
                    pixels[y * 3 + 2] = rgb & 0xFF;
                }
                break;
        }
        return pixels;
    }
    
    /**
     * Compute Mean Squared Error between two edges.
     * Lower = better match. Returns negative MSE so higher is better.
     */
    private double computeEdgeSimilarity(int[] edge1, int[] edge2) {
        int len1 = edge1.length;
        int len2 = edge2.length;
        
        // Resample edge2 to match edge1 length if needed
        int[] edge2Resized = edge2;
        if (len1 != len2) {
            edge2Resized = resampleEdgePixels(edge2, len1);
        }
        
        // Compute MSE (edges already have separate RGB channels)
        double sumSqDiff = 0.0;
        for (int i = 0; i < len1; i++) {
            int diff = edge1[i] - edge2Resized[i];
            sumSqDiff += diff * diff;
        }
        double mse = sumSqDiff / len1;
        
        return -mse;  // Negative so higher is better
    }
    
    /**
     * Resample edge pixels to a new length using nearest neighbor.
     * Input/output are flattened RGB arrays.
     */
    private int[] resampleEdgePixels(int[] edge, int newLen) {
        int[] resampled = new int[newLen];
        if (newLen <= 0) return resampled;
        
        for (int i = 0; i < newLen; i++) {
            double ratio = i / (double) (newLen - 1);
            int srcIdx = (int) Math.round(ratio * (edge.length - 1));
            srcIdx = Math.min(srcIdx, edge.length - 1);
            resampled[i] = edge[srcIdx];
        }
        
        return resampled;
    }
    
    //Check if two edges can be adjacent based on direction.
    private boolean areEdgesCompatible(EdgeType e1, EdgeType e2) {
        return (e1 == EdgeType.TOP && e2 == EdgeType.BOTTOM) ||
               (e1 == EdgeType.BOTTOM && e2 == EdgeType.TOP) ||
               (e1 == EdgeType.LEFT && e2 == EdgeType.RIGHT) ||
               (e1 == EdgeType.RIGHT && e2 == EdgeType.LEFT);
    }
    
    /**
     * Perform edge matching between all pieces using MSE-based algorithm.
     * Returns list of potential matches sorted by score (higher is better).
     */
    public List<EdgeMatch> performEdgeMatching() {
        List<EdgeMatch> matches = new ArrayList<>();
        System.out.println("Starting MSE-based edge matching...");
        
        int n = pieces.size();
        
        // For each pair of pieces, find the best matching edges
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i == j) continue;
                
                PuzzlePiece p1 = pieces.get(i);
                PuzzlePiece p2 = pieces.get(j);
                
                double bestScore = Double.NEGATIVE_INFINITY;
                EdgeType bestEdge1 = null;
                EdgeType bestEdge2 = null;
                
                // Try all edge combinations from p1
                for (EdgeType e1 : EdgeType.values()) {
                    int[] edge1Pixels = extractEdgePixels(p1.image, e1);
                    
                    // Try all edge combinations from p2
                    for (EdgeType e2 : EdgeType.values()) {
                        // Check directional compatibility
                        if (!areEdgesCompatible(e1, e2)) {
                            continue;
                        }
                        
                        int[] edge2Pixels = extractEdgePixels(p2.image, e2);
                        double score = computeEdgeSimilarity(edge1Pixels, edge2Pixels);
                        
                        if (score > bestScore) {
                            bestScore = score;
                            bestEdge1 = e1;
                            bestEdge2 = e2;
                        }
                    }
                }
                
                if (bestEdge1 != null && bestEdge2 != null) {
                    matches.add(new EdgeMatch(i, j, bestEdge1, bestEdge2, bestScore));
                }
            }
        }
        
        // Sort by score (highest first)
        Collections.sort(matches);
        
        System.out.println("Generated " + matches.size() + " potential matches");
        System.out.println("\nTop 20 matches:");
        for (int k = 0; k < Math.min(20, matches.size()); k++) {
            EdgeMatch m = matches.get(k);
            System.out.println("  " + m);
        }
        
        return matches;
    }

    // ~~~~~~~~~~~~~~~
    // Puzzle Assembly
    
    public Map<Integer, GridPos> computeLayoutFromGraph(List<EdgeMatch> matches) {
        Map<Integer, GridPos> layout = new HashMap<>();
        Set<Integer> placed = new HashSet<>();
        Set<Integer> remaining = new HashSet<>();
        
        for (PuzzlePiece p : pieces) {
            remaining.add(p.id);
        }
        
        if (matches.isEmpty()) {
            // Fallback
            layout.put(0, new GridPos(0, 0));
            placed.add(0);
            remaining.remove(0);
        } else {
            // Start with the best matching pair (highest score = smallest error)
            EdgeMatch bestPair = matches.get(0);
            layout.put(bestPair.piece1, new GridPos(0, 0));
            placed.add(bestPair.piece1);
            remaining.remove(bestPair.piece1);
            
            // Place second piece based on edge direction
            // If piece1's edge matches piece2's opposite edge,
            // piece2 is positioned relative to piece1 based on piece1's edge direction
            int piece2 = bestPair.piece2;
            GridPos piece1Pos = layout.get(bestPair.piece1);
            GridPos piece2Pos;
            
            // Calculate rotation needed for piece2 based on edge match
            int rotationNeeded = calculateRotation(bestPair.edge2, getOppositeEdge(bestPair.edge1));
            pieces.get(piece2).additionalRotation = rotationNeeded;
            
            switch (bestPair.edge1) {
                case RIGHT:
                    // piece1's right edge matches piece2's left edge
                    // piece2 is to the RIGHT of piece1
                    piece2Pos = new GridPos(piece1Pos.row, piece1Pos.col + 1);  // same row, col+1
                    break;
                case LEFT:
                    // piece1's left edge matches piece2's right edge
                    // piece2 is to the LEFT of piece1
                    piece2Pos = new GridPos(piece1Pos.row, piece1Pos.col - 1);  // same row, col-1
                    break;
                case BOTTOM:
                    // piece1's bottom edge matches piece2's top edge
                    // piece2 is BELOW piece1
                    piece2Pos = new GridPos(piece1Pos.row + 1, piece1Pos.col);  // row+1, same col
                    break;
                case TOP:
                    // piece1's top edge matches piece2's bottom edge
                    // piece2 is ABOVE piece1
                    piece2Pos = new GridPos(piece1Pos.row - 1, piece1Pos.col);  // row-1, same col
                    break;
                default:
                    piece2Pos = new GridPos(piece1Pos.row, piece1Pos.col + 1);
            }
            layout.put(piece2, piece2Pos);
            placed.add(piece2);
            remaining.remove(piece2);
            
            System.out.println("Starting with best pair: Piece " + bestPair.piece1 + 
                             " at (" + piece1Pos.row + "," + piece1Pos.col + ") and Piece " + piece2 +
                             " at (" + piece2Pos.row + "," + piece2Pos.col + ")" +
                             " [" + bestPair.edge1 + " matches " + bestPair.edge2 + ", rotation=" + rotationNeeded + "°]");
        }
        
        // Greedy placement: for each remaining piece, find best position
        while (!remaining.isEmpty()) {
            int bestTile = -1;
            GridPos bestPos = null;
            double bestScore = Double.NEGATIVE_INFINITY;
            
            for (int candidateId : remaining) {
                // Try to match with each placed piece
                for (int placedId : placed) {
                    // Check all 4 edges of the placed piece
                    for (EdgeType edge : EdgeType.values()) {
                        // Find the matching edge in candidate
                        EdgeType oppositeEdge = getOppositeEdge(edge);
                        
                        // Search for a match between these edges
                        for (EdgeMatch m : matches) {
                            if (m.piece1 == placedId && m.edge1 == edge && 
                                m.piece2 == candidateId && m.edge2 == oppositeEdge) {
                                
                                GridPos placedPos = layout.get(placedId);
                                GridPos candidatePos;
                                
                                // Calculate candidate position based on edge
                                switch (edge) {
                                    case TOP:
                                        candidatePos = new GridPos(placedPos.row - 1, placedPos.col);
                                        break;
                                    case BOTTOM:
                                        candidatePos = new GridPos(placedPos.row + 1, placedPos.col);
                                        break;
                                    case LEFT:
                                        candidatePos = new GridPos(placedPos.row, placedPos.col - 1);
                                        break;
                                    case RIGHT:
                                        candidatePos = new GridPos(placedPos.row, placedPos.col + 1);
                                        break;
                                    default:
                                        candidatePos = placedPos;
                                }
                                
                                // Check if position is free
                                boolean positionFree = true;
                                for (GridPos pos : layout.values()) {
                                    if (pos.equals(candidatePos)) {
                                        positionFree = false;
                                        break;
                                    }
                                }
                                
                                // Higher score = better match (lower error)
                                if (positionFree && m.score > bestScore) {
                                    bestScore = m.score;
                                    bestTile = candidateId;
                                    bestPos = candidatePos;
                                    // Calculate rotation needed for this piece
                                    int rotationNeeded = calculateRotation(m.edge2, oppositeEdge);
                                    pieces.get(candidateId).additionalRotation = rotationNeeded;
                                }
                            }
                        }
                    }
                }
            }
            
            // fallback
            if (bestTile == -1) {
                // Pick any remaining tile
                bestTile = remaining.iterator().next();
                System.out.println("  [FALLBACK] No match found for piece " + bestTile + ", using fallback placement");
                
                // Find first free adjacent position to any already placed piece
                GridPos newPos = null;
                outerLoop:
                for (GridPos pos : layout.values()) {
                    // Try all 4 adjacent positions
                    GridPos[] candidates = {
                        new GridPos(pos.row, pos.col + 1),   // right (prefer horizontal)
                        new GridPos(pos.row + 1, pos.col),   // below
                        new GridPos(pos.row, pos.col - 1),   // left
                        new GridPos(pos.row - 1, pos.col)    // above
                    };
                    for (GridPos cand : candidates) {
                        boolean isFree = true;
                        for (GridPos existing : layout.values()) {
                            if (existing.equals(cand)) {
                                isFree = false;
                                break;
                            }
                        }
                        if (isFree) {
                            newPos = cand;
                            break outerLoop;
                        }
                    }
                }
                // If still no position found, expand grid
                if (newPos == null) {
                    // Find max row and col
                    int maxRow = Integer.MIN_VALUE;
                    int maxCol = Integer.MIN_VALUE;
                    for (GridPos pos : layout.values()) {
                        maxRow = Math.max(maxRow, pos.row);
                        maxCol = Math.max(maxCol, pos.col);
                    }
                    newPos = new GridPos(0, maxCol + 1);
                }
                bestPos = newPos;
            } else {
                System.out.println("  Piece " + bestTile + " -> GridPos(row=" + bestPos.row + ", col=" + bestPos.col + ")" +
                                 " with score " + bestScore + " [rotation=" + pieces.get(bestTile).additionalRotation + "°]");
            }
            
            layout.put(bestTile, bestPos);
            placed.add(bestTile);
            remaining.remove(bestTile);
        }
        
        System.out.println("Placed " + placed.size() + " out of " + pieces.size() + " pieces");
        System.out.println("\nLayout computed:");
        for (int i = 0; i < pieces.size(); i++) {
            if (layout.containsKey(i)) {
                GridPos pos = layout.get(i);
                System.out.println("  Piece " + i + " -> GridPos(row=" + pos.row + ", col=" + pos.col + ")");
            }
        }
        return layout;
    }
    
    /**
     * Calculate how many degrees clockwise rotation is needed to align currentEdge to targetEdge.
     * For example, if piece's LEFT edge needs to become TOP, rotate 90° clockwise.
     */
    private int calculateRotation(EdgeType currentEdge, EdgeType targetEdge) {
        if (currentEdge == targetEdge) return 0;
        
        // Map edges to rotation values (0=TOP, 90=RIGHT, 180=BOTTOM, 270=LEFT)
        int current = edgeToRotation(currentEdge);
        int target = edgeToRotation(targetEdge);
        
        int rotation = (target - current + 360) % 360;
        return rotation;
    }
    
    private int edgeToRotation(EdgeType edge) {
        switch (edge) {
            case TOP: return 0;
            case RIGHT: return 90;
            case BOTTOM: return 180;
            case LEFT: return 270;
            default: return 0;
        }
    }
    
    private EdgeType getOppositeEdge(EdgeType edge) {
        switch (edge) {
            case TOP: return EdgeType.BOTTOM;
            case BOTTOM: return EdgeType.TOP;
            case LEFT: return EdgeType.RIGHT;
            case RIGHT: return EdgeType.LEFT;
            default: return edge;
        }
    }

    public void placeRemainingPieces(Map<Integer, GridPos> layout,
                                    List<EdgeMatch> matches) {
        
        List<Integer> remaining = new ArrayList<>();
        for (PuzzlePiece p : pieces) {
            if (!layout.containsKey(p.id)) {
                remaining.add(p.id);
            }
        }
        
        if (remaining.isEmpty()) {
            System.out.println("All pieces placed!");
            return;
        }
        
        System.out.println("Placing " + remaining.size() + " remaining pieces...");
        
        // Try to place unplaced pieces by finding any adjacent placement
        for (int pieceId : remaining) {
            int bestPlacedId = -1;
            GridPos bestPos = null;
            double bestScore = Double.NEGATIVE_INFINITY;
            
            // Try each already-placed piece as an anchor
            for (Integer placedId : layout.keySet()) {
                GridPos placedPos = layout.get(placedId);
                
                // Try all 4 directions around the placed piece
                GridPos[] neighborPositions = {
                    new GridPos(placedPos.row - 1, placedPos.col),  // TOP
                    new GridPos(placedPos.row + 1, placedPos.col),  // BOTTOM
                    new GridPos(placedPos.row, placedPos.col - 1),  // LEFT
                    new GridPos(placedPos.row, placedPos.col + 1)   // RIGHT
                };
                
                for (GridPos testPos : neighborPositions) {
                    // Check if position is free
                    boolean positionFree = true;
                    for (GridPos pos : layout.values()) {
                        if (pos.equals(testPos)) {
                            positionFree = false;
                            break;
                        }
                    }
                    
                    if (!positionFree) continue;
                    
                    // Find if there's a good match between these pieces
                    double matchScore = -1000;  // Default bad score
                    for (EdgeMatch m : matches) {
                        if (m.piece1 == placedId && m.piece2 == pieceId) {
                            matchScore = Math.max(matchScore, m.score);
                        }
                    }
                    
                    if (matchScore > bestScore) {
                        bestScore = matchScore;
                        bestPlacedId = placedId;
                        bestPos = testPos;
                    }
                }
            }
            
            if (bestPos != null) {
                layout.put(pieceId, bestPos);
            } else {
                // Fallback: place at an empty adjacent spot
                GridPos anyPlaced = layout.values().iterator().next();
                layout.put(pieceId, new GridPos(anyPlaced.row + 1, anyPlaced.col));
            }
        }
    }

    private void normalizeMLayoutAndComputeFinalPositions(Map<Integer, GridPos> layout) {
        if (layout.isEmpty()) return;
        
        // Normalize to start at (0,0)
        int minRow = Integer.MAX_VALUE, minCol = Integer.MAX_VALUE;
        int maxRow = Integer.MIN_VALUE, maxCol = Integer.MIN_VALUE;
        
        for (GridPos pos : layout.values()) {
            minRow = Math.min(minRow, pos.row);
            minCol = Math.min(minCol, pos.col);
            maxRow = Math.max(maxRow, pos.row);
            maxCol = Math.max(maxCol, pos.col);
        }
        
        System.out.println("Grid dimensions: rows [" + minRow + " to " + maxRow + "], cols [" + minCol + " to " + maxCol + "]");
        
        // Compute column widths and row heights
        Map<Integer, Integer> colWidths = new HashMap<>();
        Map<Integer, Integer> rowHeights = new HashMap<>();
        
        for (PuzzlePiece p : pieces) {
            GridPos pos = layout.get(p.id);
            if (pos != null) {
                int normalizedCol = pos.col - minCol;
                int normalizedRow = pos.row - minRow;
                
                colWidths.put(normalizedCol, 
                    Math.max(colWidths.getOrDefault(normalizedCol, 0), p.image.getWidth()));
                rowHeights.put(normalizedRow,
                    Math.max(rowHeights.getOrDefault(normalizedRow, 0), p.image.getHeight()));
                
                p.gridRow = normalizedRow;
                p.gridCol = normalizedCol;
            }
        }
        
        // Compute pixel positions for each grid cell (top-left corner)
        Map<Integer, Integer> colX = new HashMap<>();
        Map<Integer, Integer> rowY = new HashMap<>();
        
        int margin = 20;
        int curX = margin;
        for (int c = 0; c <= maxCol - minCol; c++) {
            colX.put(c, curX);
            int width = colWidths.getOrDefault(c, 100);
            curX += width;
            System.out.println("  Col " + c + ": x=" + colX.get(c) + ", width=" + width);
        }
        
        int curY = margin;
        for (int r = 0; r <= maxRow - minRow; r++) {
            rowY.put(r, curY);
            int height = rowHeights.getOrDefault(r, 100);
            curY += height;
            System.out.println("  Row " + r + ": y=" + rowY.get(r) + ", height=" + height);
        }
        
        // Set final positions
        for (PuzzlePiece p : pieces) {
            if (p.gridRow >= 0 && p.gridCol >= 0) {
                int cellW = colWidths.getOrDefault(p.gridCol, p.image.getWidth());
                int cellH = rowHeights.getOrDefault(p.gridRow, p.image.getHeight());
                
                // Center the piece in its cell
                double cx = colX.get(p.gridCol) + cellW / 2.0;
                double cy = rowY.get(p.gridRow) + cellH / 2.0;
                
                p.finalPosition = new Point((int) cx, (int) cy);
                p.finalRotation = 0;  // End state is axis-aligned
                
                System.out.println("  Piece " + p.id + " final pos: (" + (int)cx + ", " + (int)cy + ")" +
                                 " from initial: (" + p.initialPosition.x + ", " + p.initialPosition.y + ")" +
                                 " rotation: " + p.initialRotation + " -> " + p.finalRotation +
                                 " (additional: " + p.additionalRotation + "°)");
            }
        }
    }

    // ~~~~~~~~~~~~~~~
    // Animation
    
    public void showAnimatedSolution(Map<Integer, GridPos> layout) {
        normalizeMLayoutAndComputeFinalPositions(layout);
        
        int frameCount = 80;  // slower, smoother animation
        int canvasW = width;
        int canvasH = height;
        
        System.out.println("Starting animation with " + frameCount + " frames");
        
        for (int f = 0; f < frameCount; f++) {
            double t = f / (double) (frameCount - 1);
            double rotateAlpha = Math.min(1.0, t / 0.5);          // rotation completes by 50%
            double moveAlpha   = Math.max(0.0, (t - 0.3) / 0.7);   // movement ramps from 30% to 100%
            
            BufferedImage canvas = new BufferedImage(canvasW, canvasH, BufferedImage.TYPE_INT_RGB);
            Graphics2D g = canvas.createGraphics();
            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
            g.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
            g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g.setColor(Color.BLACK);
            g.fillRect(0, 0, canvasW, canvasH);
            
            for (PuzzlePiece p : pieces) {
                // Interpolate position and rotation (linear interpolation)
                double x = p.initialPosition.x * (1 - moveAlpha) + p.finalPosition.x * moveAlpha;
                double y = p.initialPosition.y * (1 - moveAlpha) + p.finalPosition.y * moveAlpha;
                double angle = p.initialRotation * (1 - rotateAlpha) + p.finalRotation * rotateAlpha;
                
                // Draw piece at interpolated position/rotation
                BufferedImage img = p.image;
                // On the final frame, use a 1px replicated border to close tiny seams
                if (f == frameCount - 1) {
                    img = replicateBorderPixels(img, 1);
                }
                
                // Apply transformation: translate to position, rotate around center
                AffineTransform transform = new AffineTransform();
                transform.translate(x, y);  // Move to current position
                transform.rotate(Math.toRadians(angle));  // Rotate around center
                transform.translate(-img.getWidth() / 2.0, -img.getHeight() / 2.0);  // Center the image
                
                g.drawImage(img, transform, null);
            }
            
            g.dispose();
            
            // Display frame
            lbIm1.setIcon(new ImageIcon(canvas));
            if (frame != null) {
                frame.repaint();
            }
            
            try {
                Thread.sleep(30);  // ~33 fps
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
        
        System.out.println("Animation complete");
    }

    public void showInputImage() {
        JFrame inputFrame = new JFrame("Input Image");
        inputFrame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        inputFrame.add(new JLabel(new ImageIcon(imgOne)));
        inputFrame.pack();
        inputFrame.setVisible(true);
    }

    public void showReconstructedPuzzle(Map<Integer, GridPos> layout) {
        normalizeMLayoutAndComputeFinalPositions(layout);
        
        // Calculate canvas size
        int maxRow = 0, maxCol = 0;
        for (PuzzlePiece p : pieces) {
            maxRow = Math.max(maxRow, p.gridRow);
            maxCol = Math.max(maxCol, p.gridCol);
        }
        
        int canvasW = (maxCol + 1) * 100;
        int canvasH = (maxRow + 1) * 100;
        BufferedImage canvas = new BufferedImage(canvasW, canvasH, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = canvas.createGraphics();
        g.setColor(Color.BLACK);
        g.fillRect(0, 0, canvasW, canvasH);
        
        for (PuzzlePiece p : pieces) {
            // close gaps
            BufferedImage img = replicateBorderPixels(p.image, 1);
            
            // Apply additional rotation if needed
            if (p.additionalRotation != 0) {
                AffineTransform transform = new AffineTransform();
                transform.translate(p.finalPosition.x, p.finalPosition.y);
                transform.rotate(Math.toRadians(p.additionalRotation));
                transform.translate(-img.getWidth() / 2.0, -img.getHeight() / 2.0);
                g.drawImage(img, transform, null);
            } else {
                int drawX = (int) p.finalPosition.x - img.getWidth() / 2;
                int drawY = (int) p.finalPosition.y - img.getHeight() / 2;
                g.drawImage(img, drawX, drawY, null);
            }
        }
        
        g.dispose();
        
        JFrame resultFrame = new JFrame("Reconstructed Puzzle");
        resultFrame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        resultFrame.add(new JLabel(new ImageIcon(canvas)));
        resultFrame.pack();
        resultFrame.setVisible(true);
    }

    // ~~~~~~~~~~~~~~~
    // Visual debug
    
    public void showPiecesRotatedOnCanvas() {
        BufferedImage display = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = display.createGraphics();
        g.setColor(Color.BLACK);
        g.fillRect(0, 0, width, height);

        for (PuzzlePiece piece : pieces) {
            BufferedImage rotated = piece.image;
            int px = piece.bounds.x;
            int py = piece.bounds.y;
            int drawX = px + (piece.bounds.width - rotated.getWidth()) / 2;
            int drawY = py + (piece.bounds.height - rotated.getHeight()) / 2;
            g.drawImage(rotated, drawX, drawY, null);
        }

        g.dispose();
        lbIm1 = new JLabel(new ImageIcon(display));
    }

    public void showExtractedPieces() {
        if (pieces.isEmpty()) {
            System.out.println("No pieces detected!");
            return;
        }

        JFrame pieceFrame = new JFrame("Extracted Pieces");
        pieceFrame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);

        JPanel panel = new JPanel();
        panel.setLayout(new GridLayout(0, 4, 10, 10));

        for (PuzzlePiece piece : pieces) {
            JLabel label = new JLabel(new ImageIcon(piece.image));
            label.setBorder(BorderFactory.createTitledBorder("Piece " + piece.id));
            panel.add(label);
        }

        JScrollPane scrollPane = new JScrollPane(panel);
        pieceFrame.add(scrollPane);
        pieceFrame.setSize(900, 700);
        pieceFrame.setVisible(true);
    }

    public void showIms() {
        frame = new JFrame();
        GridBagLayout gLayout = new GridBagLayout();
        frame.getContentPane().setLayout(gLayout);

        GridBagConstraints c = new GridBagConstraints();
        c.gridx = 0;
        c.gridy = 0;

        frame.getContentPane().add(lbIm1, c);

        frame.pack();

        int animW = Math.min(width, 700);
        int animH = Math.min(height, 700);
        frame.setSize(animW, animH);
        frame.setVisible(true);
    }

    public static void main(String[] args) {
        ImageDisplay iq = new ImageDisplay();
        iq.parseArgs(args);

        iq.imgOne = new BufferedImage(iq.width, iq.height, BufferedImage.TYPE_INT_RGB);
        iq.readImageRGB(iq.width, iq.height, iq.imagePath, iq.imgOne);

        iq.detectPieces();

        // iq.showInputImage();

        iq.showPiecesRotatedOnCanvas();
        iq.showIms();
        // iq.showExtractedPieces();

        // Perform edge matching and assembly
        List<EdgeMatch> matches = iq.performEdgeMatching();
        Map<Integer, GridPos> layout = iq.computeLayoutFromGraph(matches);
        iq.placeRemainingPieces(layout, matches);

        // iq.showReconstructedPuzzle(layout);
        iq.showAnimatedSolution(layout);
    }
}
