import java.awt.*;
import java.awt.image.*;
import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import javax.swing.*;

public class ImageDisplay {
    JFrame frame;
    JLabel lbIm1;
    BufferedImage imgOne;
    int width;
    int height;

    String imagePath;

    // Detected puzzle pieces
    List<PuzzlePiece> pieces = new ArrayList<>();

    // Class to represent a detected puzzle piece
    static class PuzzlePiece {
        BufferedImage image;   // final, axis-aligned, rectangular tile
        Rectangle bounds;      // original bounding box in source image
        int id;
        double rotationAngle;  // angle (in degrees) applied to make it axis-aligned

        PuzzlePiece(BufferedImage image, Rectangle bounds, int id) {
            this.image = image;
            this.bounds = bounds;
            this.id = id;
            this.rotationAngle = 0;
        }
    }

    // -------------------------------------------------------------------------
    // Basic I/O
    // -------------------------------------------------------------------------
    public void readImageRGB(int width, int height, String imgPath, BufferedImage img) {
        try {
            int frameLength = width * height * 3;
            File file = new File(imgPath);
            RandomAccessFile raf = new RandomAccessFile(file, "r");
            raf.seek(0);
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
            raf.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void parseArgs(String[] args) {
        imagePath = args[0];
        width = Integer.parseInt(args[1]);
        height = Integer.parseInt(args[2]);
    }

    // -------------------------------------------------------------------------
    // Piece detection
    // -------------------------------------------------------------------------

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

        double rotationAngle = detectRotation(localPixels);
        BufferedImage pieceImage = extractAndRotatePiece(localPixels, bounds, rotationAngle);

        PuzzlePiece piece = new PuzzlePiece(pieceImage, bounds, id);
        piece.rotationAngle = rotationAngle;
        return piece;
    }

    /**
     * Brute-force orientation finder: for angles in [-89, 89],
     * rotate all local pixels and choose the angle that minimizes
     * the area of the bounding rectangle. This tends to align
     * the piece edges with the axes, which is what we want.
     *
     * NOTE: we return the best angle directly (no snapping).
     */
    private double detectRotation(List<Point> pixels) {
        int n = pixels.size();
        double[] xs = new double[n];
        double[] ys = new double[n];

        for (int i = 0; i < n; i++) {
            xs[i] = pixels.get(i).x;
            ys[i] = pixels.get(i).y;
        }

        double bestAngle = 0.0;
        double bestArea = Double.MAX_VALUE;

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
            }
        }

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

        BufferedImage cropped = cropToContent(rotated);
        return argbToRgb(cropped);
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

    // -------------------------------------------------------------------------
    // Visual debug
    // -------------------------------------------------------------------------
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

    // -------------------------------------------------------------------------
    // Edge matching: RGB + gradient histograms from CLEAN inner strips
    // -------------------------------------------------------------------------

    enum EdgeType {
        TOP, BOTTOM, LEFT, RIGHT
    }

    static class EdgeMatch implements Comparable<EdgeMatch> {
        int piece1Id;
        EdgeType edge1;
        int piece2Id;
        EdgeType edge2;
        double score;   // lower = better

        EdgeMatch(int p1, EdgeType e1, int p2, EdgeType e2, double score) {
            this.piece1Id = p1;
            this.edge1 = e1;
            this.piece2Id = p2;
            this.edge2 = e2;
            this.score = score;
        }

        @Override
        public int compareTo(EdgeMatch other) {
            return Double.compare(this.score, other.score);
        }

        @Override
        public String toString() {
            return String.format("Piece %d (%s) <-> Piece %d (%s): %.4f",
                    piece1Id, edge1, piece2Id, edge2, score);
        }
    }

    // Histogram configuration: 32 bins per channel => 96 total
    private static final int HIST_BINS = 32;

    // For inner-strip sampling
    private static final int EDGE_STRIP_WIDTH = 4;      // pixels inward from border
    private static final int BG_BRIGHTNESS_THRESH = 10; // "near black" threshold

    private boolean isBackgroundColor(int rgb) {
        int r = (rgb >> 16) & 0xFF;
        int g = (rgb >> 8) & 0xFF;
        int b = rgb & 0xFF;
        int bright = (r + g + b) / 3;
        return bright < BG_BRIGHTNESS_THRESH;
    }

    /**
     * Sample a CLEAN edge strip:
     *  - For each position along the edge, look inward up to EDGE_STRIP_WIDTH pixels
     *  - Ignore near-black pixels, average the rest
     *  - If everything is background, fall back to the border pixel
     */
    private int[] sampleCleanEdge(BufferedImage img, EdgeType edge) {
        int w = img.getWidth();
        int h = img.getHeight();

        if (w <= 0 || h <= 0) {
            return new int[0];
        }

        int len = (edge == EdgeType.TOP || edge == EdgeType.BOTTOM) ? w : h;
        int[] strip = new int[len];

        for (int i = 0; i < len; i++) {
            int sumR = 0, sumG = 0, sumB = 0, count = 0;

            if (edge == EdgeType.TOP) {
                int x = i;
                for (int dy = 0; dy < EDGE_STRIP_WIDTH && dy < h; dy++) {
                    int y = dy;
                    int rgb = img.getRGB(x, y);
                    if (!isBackgroundColor(rgb)) {
                        sumR += (rgb >> 16) & 0xFF;
                        sumG += (rgb >> 8) & 0xFF;
                        sumB += rgb & 0xFF;
                        count++;
                    }
                }
                if (count == 0) {
                    strip[i] = img.getRGB(x, 0);
                } else {
                    strip[i] = ((sumR / count) << 16) | ((sumG / count) << 8) | (sumB / count);
                }
            } else if (edge == EdgeType.BOTTOM) {
                int x = i;
                for (int dy = 0; dy < EDGE_STRIP_WIDTH && dy < h; dy++) {
                    int y = h - 1 - dy;
                    int rgb = img.getRGB(x, y);
                    if (!isBackgroundColor(rgb)) {
                        sumR += (rgb >> 16) & 0xFF;
                        sumG += (rgb >> 8) & 0xFF;
                        sumB += rgb & 0xFF;
                        count++;
                    }
                }
                if (count == 0) {
                    strip[i] = img.getRGB(x, h - 1);
                } else {
                    strip[i] = ((sumR / count) << 16) | ((sumG / count) << 8) | (sumB / count);
                }
            } else if (edge == EdgeType.LEFT) {
                int y = i;
                for (int dx = 0; dx < EDGE_STRIP_WIDTH && dx < w; dx++) {
                    int x = dx;
                    int rgb = img.getRGB(x, y);
                    if (!isBackgroundColor(rgb)) {
                        sumR += (rgb >> 16) & 0xFF;
                        sumG += (rgb >> 8) & 0xFF;
                        sumB += rgb & 0xFF;
                        count++;
                    }
                }
                if (count == 0) {
                    strip[i] = img.getRGB(0, y);
                } else {
                    strip[i] = ((sumR / count) << 16) | ((sumG / count) << 8) | (sumB / count);
                }
            } else { // RIGHT
                int y = i;
                for (int dx = 0; dx < EDGE_STRIP_WIDTH && dx < w; dx++) {
                    int x = w - 1 - dx;
                    int rgb = img.getRGB(x, y);
                    if (!isBackgroundColor(rgb)) {
                        sumR += (rgb >> 16) & 0xFF;
                        sumG += (rgb >> 8) & 0xFF;
                        sumB += rgb & 0xFF;
                        count++;
                    }
                }
                if (count == 0) {
                    strip[i] = img.getRGB(w - 1, y);
                } else {
                    strip[i] = ((sumR / count) << 16) | ((sumG / count) << 8) | (sumB / count);
                }
            }
        }

        return strip;
    }

    /**
     * Compute RGB histogram of a CLEAN edge strip.
     */
    private float[] computeEdgeHistogram(BufferedImage img, EdgeType edge) {
        float[] hist = new float[HIST_BINS * 3];
        Arrays.fill(hist, 0f);

        int[] strip = sampleCleanEdge(img, edge);
        if (strip.length == 0) return hist;

        for (int rgb : strip) {
            accumulateColorToHist(rgb, hist);
        }

        // normalize
        for (int i = 0; i < hist.length; i++) {
            hist[i] /= (float) strip.length;
        }

        return hist;
    }

    private void accumulateColorToHist(int rgb, float[] hist) {
        int r = (rgb >> 16) & 0xFF;
        int g = (rgb >> 8) & 0xFF;
        int b = rgb & 0xFF;

        int rBin = (r * HIST_BINS) / 256;
        int gBin = (g * HIST_BINS) / 256;
        int bBin = (b * HIST_BINS) / 256;

        if (rBin >= HIST_BINS) rBin = HIST_BINS - 1;
        if (gBin >= HIST_BINS) gBin = HIST_BINS - 1;
        if (bBin >= HIST_BINS) bBin = HIST_BINS - 1;

        hist[rBin] += 1f;
        hist[HIST_BINS + gBin] += 1f;
        hist[HIST_BINS * 2 + bBin] += 1f;
    }

    /**
     * L1 distance between two normalized histograms.
     * Lower distance = better match.
     */
    private double histogramDistance(float[] h1, float[] h2) {
        double d = 0.0;
        for (int i = 0; i < h1.length; i++) {
            d += Math.abs(h1[i] - h2[i]);
        }
        return d;
    }

    private EdgeType getComplementaryEdge(EdgeType e) {
        switch (e) {
            case TOP: return EdgeType.BOTTOM;
            case BOTTOM: return EdgeType.TOP;
            case LEFT: return EdgeType.RIGHT;
            case RIGHT: return EdgeType.LEFT;
            default: return null;
        }
    }

    //---------------------------------------------------------------
    // GRADIENT HISTOGRAMS (32 bins for gradient magnitude)
    //---------------------------------------------------------------

    private static final int GRAD_BINS = 32;

    private float[] computeGradientHistogram(BufferedImage img, EdgeType edge) {
        float[] hist = new float[GRAD_BINS];
        Arrays.fill(hist, 0f);

        int[] strip = sampleCleanEdge(img, edge);
        if (strip.length < 2) return hist;

        // 1D gradient along edge
        for (int i = 1; i < strip.length - 1; i++) {
            int c1 = strip[i - 1];
            int c2 = strip[i + 1];

            int gray1 = ((c1 >> 16) & 255) + ((c1 >> 8) & 255) + (c1 & 255);
            int gray2 = ((c2 >> 16) & 255) + ((c2 >> 8) & 255) + (c2 & 255);

            int diff = Math.abs(gray2 - gray1);  // approximate gradient magnitude
            int bin = (diff * GRAD_BINS) / 765;  // 765 = 255*3

            if (bin >= GRAD_BINS) bin = GRAD_BINS - 1;
            hist[bin]++;
        }

        // normalize
        float total = 0;
        for (float f : hist) total += f;
        if (total > 0) {
            for (int i = 0; i < GRAD_BINS; i++) hist[i] /= total;
        }

        return hist;
    }

    private double gradientDistance(float[] g1, float[] g2) {
        double d = 0;
        for (int i = 0; i < GRAD_BINS; i++) {
            d += Math.abs(g1[i] - g2[i]);
        }
        return d;
    }

    //--------------------------------------------------------------------
    // Combined RGB + Gradient Histogram Edge Matching (using clean edges)
    //--------------------------------------------------------------------
    public List<EdgeMatch> performEdgeMatching() {
        List<EdgeMatch> matches = new ArrayList<>();
        System.out.println("Starting edge matching (clean RGB + Gradient)…");

        for (int i = 0; i < pieces.size(); i++) {
            for (int j = i + 1; j < pieces.size(); j++) {

                PuzzlePiece p1 = pieces.get(i);
                PuzzlePiece p2 = pieces.get(j);

                for (EdgeType e1 : EdgeType.values()) {
                    EdgeType e2 = getComplementaryEdge(e1);

                    // --- compute RGB histograms ---
                    float[] rgb1 = computeEdgeHistogram(p1.image, e1);
                    float[] rgb2 = computeEdgeHistogram(p2.image, e2);
                    double rgbDist = histogramDistance(rgb1, rgb2);

                    // --- compute gradient histograms ---
                    float[] grad1 = computeGradientHistogram(p1.image, e1);
                    float[] grad2 = computeGradientHistogram(p2.image, e2);
                    double gradDist = gradientDistance(grad1, grad2);

                    // --- combined score ---
                    double finalScore = 0.4 * rgbDist + 0.6 * gradDist;

                    matches.add(new EdgeMatch(
                            p1.id, e1,
                            p2.id, e2,
                            finalScore
                    ));
                }
            }
        }

        Collections.sort(matches);

        System.out.println("Top 10 combined matches:");
        for (int k = 0; k < Math.min(10, matches.size()); k++) {
            System.out.println(matches.get(k));
        }

        return matches;
    }

    // -------------------------------------------------------------------------
    // Build adjacency graph (mutual-best) and layout
    // -------------------------------------------------------------------------

    static class PlacementEdge {
        int fromPiece, toPiece;
        EdgeType direction;
        double score;

        PlacementEdge(int from, int to, EdgeType dir, double score) {
            this.fromPiece = from;
            this.toPiece = to;
            this.direction = dir;
            this.score = score;
        }
    }

    public Map<Integer, Map<EdgeType, PlacementEdge>> buildAdjacencyGraph(List<EdgeMatch> matches) {

        Map<Integer, Map<EdgeType, EdgeMatch>> bestFrom = new HashMap<>();
        Map<Integer, Map<EdgeType, EdgeMatch>> bestTo = new HashMap<>();
        for (PuzzlePiece p : pieces) {
            bestFrom.put(p.id, new EnumMap<>(EdgeType.class));
            bestTo.put(p.id, new EnumMap<>(EdgeType.class));
        }

        // find best outgoing (from) and incoming (to) match for each edge
        for (EdgeMatch m : matches) {
            Map<EdgeType, EdgeMatch> outMap = bestFrom.get(m.piece1Id);
            EdgeMatch curOut = outMap.get(m.edge1);
            if (curOut == null || m.score < curOut.score) {
                outMap.put(m.edge1, m);
            }

            Map<EdgeType, EdgeMatch> inMap = bestTo.get(m.piece2Id);
            EdgeMatch curIn = inMap.get(m.edge2);
            if (curIn == null || m.score < curIn.score) {
                inMap.put(m.edge2, m);
            }
        }

        Map<Integer, Map<EdgeType, PlacementEdge>> graph = new HashMap<>();
        for (PuzzlePiece p : pieces) {
            graph.put(p.id, new EnumMap<>(EdgeType.class));
        }

        // keep only mutual best matches
        for (EdgeMatch m : matches) {
            if (bestFrom.get(m.piece1Id).get(m.edge1) == m &&
                bestTo.get(m.piece2Id).get(m.edge2) == m) {

                graph.get(m.piece1Id).put(
                        m.edge1,
                        new PlacementEdge(m.piece1Id, m.piece2Id, m.edge1, m.score));

                graph.get(m.piece2Id).put(
                        m.edge2,
                        new PlacementEdge(m.piece2Id, m.piece1Id, m.edge2, m.score));
            }
        }

        return graph;
    }

    static class GridPos {
        int row, col;
        GridPos(int r, int c) { row = r; col = c; }
    }

    public Map<Integer, GridPos> computeLayoutFromGraph(
            Map<Integer, Map<EdgeType, PlacementEdge>> graph) {

        Map<Integer, GridPos> pos = new HashMap<>();
        Queue<Integer> q = new LinkedList<>();

        if (pieces.isEmpty()) return pos;

        // use piece 0 as root at (0,0)
        pos.put(0, new GridPos(0, 0));
        q.add(0);

        while (!q.isEmpty()) {
            int cur = q.poll();
            GridPos curPos = pos.get(cur);

            Map<EdgeType, PlacementEdge> neighbors = graph.get(cur);
            if (neighbors == null) continue;

            for (Map.Entry<EdgeType, PlacementEdge> entry : neighbors.entrySet()) {
                EdgeType dir = entry.getKey();
                PlacementEdge edge = entry.getValue();

                int np = edge.toPiece;
                if (pos.containsKey(np)) continue;

                int nr = curPos.row;
                int nc = curPos.col;

                switch (dir) {
                    case TOP:    nr--; break;
                    case BOTTOM: nr++; break;
                    case LEFT:   nc--; break;
                    case RIGHT:  nc++; break;
                }

                pos.put(np, new GridPos(nr, nc));
                q.add(np);
            }
        }

        if (!pos.isEmpty()) {
            int minR = Integer.MAX_VALUE, minC = Integer.MAX_VALUE;
            for (GridPos gp : pos.values()) {
                minR = Math.min(minR, gp.row);
                minC = Math.min(minC, gp.col);
            }
            for (GridPos gp : pos.values()) {
                gp.row -= minR;
                gp.col -= minC;
            }
        }

        return pos;
    }

    public void showReconstructedPuzzle(Map<Integer, GridPos> layout) {

        if (layout.isEmpty()) {
            System.out.println("No connected component found; nothing to display.");
            return;
        }

        List<PuzzlePiece> placed = new ArrayList<>();
        for (PuzzlePiece p : pieces) {
            if (layout.containsKey(p.id)) {
                placed.add(p);
            }
        }

        if (placed.isEmpty()) {
            System.out.println("No pieces placed in layout.");
            return;
        }

        int maxR = 0, maxC = 0;
        for (GridPos gp : layout.values()) {
            maxR = Math.max(maxR, gp.row);
            maxC = Math.max(maxC, gp.col);
        }

        int tileW = 0, tileH = 0;
        for (PuzzlePiece p : placed) {
            tileW += p.image.getWidth();
            tileH += p.image.getHeight();
        }
        tileW /= placed.size();
        tileH /= placed.size();

        int canvasW = (maxC + 1) * tileW;
        int canvasH = (maxR + 1) * tileH;

        BufferedImage assembled = new BufferedImage(canvasW, canvasH, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = assembled.createGraphics();
        g.setColor(Color.BLACK);
        g.fillRect(0, 0, canvasW, canvasH);

        for (PuzzlePiece p : placed) {
            GridPos gp = layout.get(p.id);
            int x = gp.col * tileW;
            int y = gp.row * tileH;
            g.drawImage(p.image, x, y, null);
        }

        g.dispose();

        JFrame f = new JFrame("Assembled Puzzle");
        f.add(new JLabel(new ImageIcon(assembled)));
        f.pack();
        f.setVisible(true);
    }

    // -------------------------------------------------------------------------
    // Boilerplate window
    // -------------------------------------------------------------------------
    public void showIms() {
        frame = new JFrame();
        GridBagLayout gLayout = new GridBagLayout();
        frame.getContentPane().setLayout(gLayout);

        GridBagConstraints c = new GridBagConstraints();
        c.gridx = 0;
        c.gridy = 0;

        frame.getContentPane().add(lbIm1, c);

        frame.pack();
        frame.setVisible(true);
    }

    // -------------------------------------------------------------------------
    // main
    // -------------------------------------------------------------------------
    public static void main(String[] args) {
        ImageDisplay iq = new ImageDisplay();
        iq.parseArgs(args);

        iq.imgOne = new BufferedImage(iq.width, iq.height, BufferedImage.TYPE_INT_RGB);
        iq.readImageRGB(iq.width, iq.height, iq.imagePath, iq.imgOne);

        iq.detectPieces();

        iq.showPiecesRotatedOnCanvas();
        iq.showIms();
        iq.showExtractedPieces();

        List<EdgeMatch> matches = iq.performEdgeMatching();
        Map<Integer, Map<EdgeType, PlacementEdge>> graph = iq.buildAdjacencyGraph(matches);
        Map<Integer, GridPos> layout = iq.computeLayoutFromGraph(graph);

        iq.showReconstructedPuzzle(layout);
    }
}
