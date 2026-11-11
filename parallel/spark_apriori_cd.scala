// Count Distribution Apriori. Baskets are cached between rounds.
// Each round counts the candidates for this process, then the counts are summed.
// This shell is local[*] unless SPARK_MASTER says otherwise. It is not a YARN job.
// Basket files are one integer mask per line, same bits as apriori_cd_benchmark.py.

def asFileUrl(path: String): String = {
  if (path.startsWith("file:")) path
  else {
    val abs = new java.io.File(path).getAbsolutePath.replace("\\", "/")
    if (abs.startsWith("/")) "file://" + abs else "file:///" + abs
  }
}

def generateCandidates(prev: Array[Int], k: Int): Array[Int] = {
  val prevSet = prev.toSet
  val found = scala.collection.mutable.LinkedHashSet[Int]()
  var i = 0
  while (i < prev.length) {
    var j = i + 1
    while (j < prev.length) {
      val union = prev(i) | prev(j)
      val bits = Integer.bitCount(union)
      if (bits == k && !found.contains(union)) {
        var ok = true
        var b = 0
        while (b < 11 && ok) {
          val bit = 1 << b
          if ((union & bit) != 0 && !prevSet.contains(union ^ bit)) ok = false
          b += 1
        }
        if (ok) found += union
      }
      j += 1
    }
    i += 1
  }
  found.toArray
}

val inPath = asFileUrl(sys.env.getOrElse("APRIORI_IN", ""))
val outPath = sys.env.getOrElse("APRIORI_OUT", "")
val tag = sys.env.getOrElse("APRIORI_TAG", "")
val minSupport = sys.env.getOrElse("APRIORI_MIN_SUPPORT", "0.05").toDouble
val maxK = sys.env.getOrElse("APRIORI_MAX_K", "3").toInt

val tLoad0 = System.nanoTime()
val data = sc.textFile(inPath).map(_.trim).filter(_.nonEmpty).map(_.toInt).cache()
val n = data.count()
val loadSec = (System.nanoTime() - tLoad0) / 1e9
val minCount = math.max(1L, math.ceil(minSupport * n.toDouble).toLong)

var prev = Array.empty[Int]
val frequent = scala.collection.mutable.LinkedHashMap[Int, Long]()
var rounds = 0
var candidatesCounted = 0
val tCount0 = System.nanoTime()
var k = 1
while (k <= maxK) {
  val cands = if (k == 1) (0 until 11).map(i => 1 << i).toArray else generateCandidates(prev, k)
  if (cands.isEmpty) k = maxK + 1
  else {
    val bc = sc.broadcast(cands)
    val counts = data.mapPartitions(iter => {
      val cs = bc.value
      val acc = Array.fill(cs.length)(0L)
      while (iter.hasNext) {
        val mask = iter.next()
        var i = 0
        while (i < cs.length) {
          val cand = cs(i)
          if ((mask & cand) == cand) acc(i) += 1L
          i += 1
        }
      }
      Iterator.single(acc)
    }).reduce((a, b) => {
      var i = 0
      val out = new Array[Long](a.length)
      while (i < a.length) {
        out(i) = a(i) + b(i)
        i += 1
      }
      out
    })
    bc.destroy()
    val next = scala.collection.mutable.ArrayBuffer[Int]()
    var i = 0
    while (i < cands.length) {
      if (counts(i) >= minCount) {
        frequent(cands(i)) = counts(i)
        next += cands(i)
      }
      i += 1
    }
    prev = next.toArray
    rounds += 1
    candidatesCounted += cands.length
    if (prev.isEmpty) k = maxK + 1
  }
  k += 1
}
val countSec = (System.nanoTime() - tCount0) / 1e9
val wallSec = loadSec + countSec

val outFile = new java.io.File(outPath)
val parent = outFile.getParentFile
if (parent != null && !parent.exists()) parent.mkdirs()
val pw = new java.io.PrintWriter(outFile)
pw.println("engine=Spark")
pw.println("algorithm=Apriori-Count-Distribution")
pw.println("status=ok")
pw.println("tag=" + tag)
pw.println("baskets=" + n)
pw.println("wall_seconds=" + wallSec)
pw.println("load_seconds=" + loadSec)
pw.println("count_seconds=" + countSec)
pw.println("rounds=" + rounds)
pw.println("candidates_counted=" + candidatesCounted)
pw.println("min_count=" + minCount)
pw.println("min_support=" + minSupport)
pw.println("max_itemset=" + maxK)
pw.println("workers=" + sc.defaultParallelism)
pw.println("spark_master=" + sc.master)
pw.println("data=" + inPath)
frequent.toSeq.sortBy(_._1).foreach { case (mask, count) =>
  pw.println("freq=" + mask + "\t" + count)
}
pw.close()
println("Spark Apriori " + tag + " baskets=" + n + " wall_seconds=" + wallSec + " master=" + sc.master)

if (sys.env.getOrElse("APRIORI_BATCH", "") == "1") {
  System.exit(0)
}
