#!/usr/bin/env bash
export HOME="${HOME:-/home/bitnami}"
# shellcheck disable=SC1090
source "$HOME/.bashrc" 2>/dev/null || true
export HADOOP_HOME="${HADOOP_HOME:-/usr/local/hadoop-2.6.0}"
export HADOOP_CONF_DIR="${HADOOP_CONF_DIR:-$HADOOP_HOME/etc/hadoop}"
export SPARK_HOME="${SPARK_HOME:-/usr/local/spark-1.2.2-bin-hadoop2.4}"
export JAVA_HOME="${JAVA_HOME:-/usr/local/jdk1.8.0_40}"
export PATH="$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$SPARK_HOME/bin:$JAVA_HOME/bin:/bin:/usr/bin:/usr/local/bin:$PATH"
export STREAMING_JAR="${STREAMING_JAR:-$HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-2.6.0.jar}"
# Shared folder mount point for this project
export HOUSING="${HOUSING:-/home/bitnami/uk_housing}"
