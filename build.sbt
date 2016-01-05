organization := "com.github.nearbydelta"

name := "DeepSpark"

version := "1.2.0"

scalaVersion := "2.10.6"

scalacOptions += "-target:jvm-1.7"

crossScalaVersions := Seq("2.10.6", "2.11.7")

resolvers ++= Seq("snapshots", "releases").map(Resolver.sonatypeRepo)

resolvers ++= Seq(
  "Typesafe Releases" at "http://repo.typesafe.com/typesafe/releases/"
)

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "1.5.1",
  "org.apache.spark" %% "spark-mllib" % "1.5.1"
)

scalacOptions in Test ++= Seq("-Yrangepos")

licenses := Seq("GNU General Public License v2" → url("http://www.gnu.org/licenses/gpl-2.0.html"))

homepage := Some(url("http://nearbydelta.github.io/DeepSpark"))

publishTo <<= version { v: String ⇒
  val nexus = "https://oss.sonatype.org/"
  if (v.trim.endsWith("SNAPSHOT"))
    Some("snapshots" at nexus + "content/repositories/snapshots")
  else
    Some("releases" at nexus + "service/local/staging/deploy/maven2")
}

publishMavenStyle := true

publishArtifact in Test := false

pomIncludeRepository := { x ⇒ false}

pomExtra :=
  <scm>
    <url>git@github.com:nearbydelta/DeepSpark.git</url>
    <connection>scm:git:git@github.com:nearbydelta/DeepSpark.git</connection>
  </scm>
    <developers>
      <developer>
        <id>nearbydelta</id>
        <name>Bugeun Kim</name>
        <url>http://bydelta.kr</url>
      </developer>
    </developers>
