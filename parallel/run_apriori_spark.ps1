# Time Spark local[*] Count Distribution on the basket files written by
# local_python/apriori_cd_benchmark.py. This is not a YARN job.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$SparkHome = "C:\cs4480_tools\spark-3.5.3-bin-hadoop3"
$JavaHome = "C:\Program Files\Eclipse Adoptium\jdk-17.0.21.9-hotspot"
$HadoopHome = "C:\cs4480_tools\winutils"
$env:JAVA_HOME = $JavaHome
$env:SPARK_HOME = $SparkHome
$env:HADOOP_HOME = $HadoopHome
$env:PATH = "$SparkHome\bin;$JavaHome\bin;$HadoopHome\bin;$env:PATH"
$env:APRIORI_BATCH = "1"
$env:APRIORI_MIN_SUPPORT = "0.05"
$env:APRIORI_MAX_K = "3"

$HostName = $env:COMPUTERNAME
if (-not $HostName) { $HostName = "goodpc" }
$SafeHost = ($HostName -replace "[^A-Za-z0-9._-]", "_")
$OutDir = Join-Path $Root "data\processed\apriori_timing\$SafeHost"
$BasketDir = Join-Path $Root "data\parallel\apriori_baskets"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
New-Item -ItemType Directory -Force -Path "C:\cs4480_data\spark_tmp" | Out-Null

$shell = Join-Path $SparkHome "bin\spark-shell.cmd"
if (-not (Test-Path $shell)) {
    Write-Output "spark-shell missing: $shell"
    exit 1
}

$tags = @("n200000", "n1000000", "n5000000", "n10000000", "n20000000", "all")
foreach ($tag in $tags) {
    $data = Join-Path $BasketDir "$tag.txt"
    $out = Join-Path $OutDir "spark_$tag.txt"
    if (-not (Test-Path $data)) {
        Write-Output "missing baskets $data"
        exit 1
    }
    Remove-Item -Force $out -ErrorAction SilentlyContinue
    $env:APRIORI_IN = $data
    $env:APRIORI_OUT = $out
    $env:APRIORI_TAG = $tag
    Write-Output "Spark $tag"
    & $shell --master "local[*]" --driver-memory 8g --conf "spark.ui.enabled=false" --conf "spark.local.dir=C:/cs4480_data/spark_tmp" --conf "spark.hadoop.io.native.lib.available=false" -i (Join-Path $Root "parallel\spark_apriori_cd.scala")
    if (-not (Test-Path $out)) {
        Write-Output "Spark did not write $out"
        exit 1
    }
    & "C:\Users\jkjk2\AppData\Local\Programs\Python\Python313\python.exe" (Join-Path $Root "parallel\machine_info.py") --append $out
    Add-Content -Path $out -Value "spark_master_requested=local[*]"
    Add-Content -Path $out -Value "cpu_model_reported=AMD Ryzen 5 5600X 6-Core Processor"
    Add-Content -Path $out -Value "cpu_cores_physical=6"
    Add-Content -Path $out -Value "cpu_threads=12"
    Add-Content -Path $out -Value "mem_total_gb_reported=31.889"
}
Write-Output "Spark results: $OutDir"
