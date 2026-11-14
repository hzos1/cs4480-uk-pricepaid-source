// One Spark local[12] session. The same Lloyd stitch, with 6, 12, 24, 36,
// 48, and 60 contiguous pieces. Each count is its own stage.

import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Paths

val rawPath = sys.env("STITCH_RAW")
val centroidPath = sys.env("STITCH_CENTROIDS")
val pieceScript = sys.env("STITCH_SCRIPT")
val pythonExe = sys.env("STITCH_PYTHON")
val maxIter = sys.env.getOrElse("KMEANS_MAX_ITER", "300")
val workRoot = sys.env.getOrElse("STITCH_WORK", "C:/cs4480_data/spark-tmp/stitch-sweep")
val outDir = sys.env("STITCH_OUT_DIR")
val sweep = sys.env.getOrElse("STITCH_SWEEP", "6,12,24,36,48,60").split(",").map(_.trim).filter(_.nonEmpty).map(_.toInt)

val rawFile = new java.io.File(rawPath)
val n = (rawFile.length() / (6 * 8)).toInt
if (n <= 0) sys.error("empty matrix " + rawPath)

def chunkRanges(rows: Int, pieces: Int): Array[(Int, Int)] = {
  val p = math.max(1, math.min(pieces, rows))
  val base = rows / p
  val rem = rows % p
  val out = new Array[(Int, Int)](p)
  var start = 0
  var i = 0
  while (i < p) {
    val size = base + (if (i < rem) 1 else 0)
    out(i) = (start, size)
    start += size
    i += 1
  }
  out
}

def runPieces(pieceCount: Int, outPath: String): Double = {
  val ranges = chunkRanges(n, pieceCount)
  val workDir = new java.io.File(workRoot, "p" + ranges.length)
  workDir.mkdirs()
  val t0 = System.nanoTime()
  val lines = sc.parallelize(ranges.indices, ranges.length).map { piece =>
    val start = ranges(piece)._1
    val rows = ranges(piece)._2
    val outFile = new java.io.File(workDir, "piece_" + piece + ".json")
    val errFile = new java.io.File(workDir, "piece_" + piece + ".err")
    val proc = new ProcessBuilder(
      pythonExe,
      pieceScript,
      "--raw", rawPath,
      "--start", start.toString,
      "--rows", rows.toString,
      "--centroids", centroidPath,
      "--piece", piece.toString,
      "--max-iter", maxIter
    ).redirectOutput(outFile).redirectError(errFile).start()
    val code = proc.waitFor()
    if (code != 0) {
      val err = new String(Files.readAllBytes(errFile.toPath), StandardCharsets.UTF_8)
      sys.error("piece " + piece + " exit " + code + " " + err.take(2000))
    }
    new String(Files.readAllBytes(outFile.toPath), StandardCharsets.UTF_8).trim
  }.collect()
  val stageSec = (System.nanoTime() - t0) / 1e9
  val sorted = lines.sortBy { line =>
    val key = "\"piece\": "
    val at = line.indexOf(key)
    if (at < 0) 0 else line.substring(at + key.length).takeWhile(_.isDigit).toInt
  }
  val writer = Files.newBufferedWriter(Paths.get(outPath), StandardCharsets.UTF_8)
  try {
    writer.write("stage_seconds=%.6f\n".format(stageSec))
    writer.write("points=%d\n".format(n))
    writer.write("pieces=%d\n".format(ranges.length))
    writer.write("spark_master=%s\n".format(sc.master))
    sorted.foreach { line =>
      writer.write(line)
      writer.write("\n")
    }
  } finally writer.close()
  println("pieces=%d stage_seconds=%.3f wrote %s".format(ranges.length, stageSec, outPath))
  stageSec
}

new java.io.File(outDir).mkdirs()
sweep.foreach { pieceCount =>
  val outPath = outDir + "/stitch_cpu_p%d_pieces.txt".format(pieceCount)
  runPieces(pieceCount, outPath)
}

println("sweep_done")
if (sys.env.getOrElse("KMEANS_BATCH", "") == "1") {
  System.exit(0)
}
