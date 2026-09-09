// One JSON file per CPG: every real method with what the backend convention needs.
//
// Condition polarity mirrors backend_ts._enclosing_conditions, so the two backends
// disagree about symbol and call RESOLUTION -- the experimental variable -- and never
// about what a guard means: a call inside a structure's own condition is not guarded by
// it, and a call reached through an else branch runs when the condition is FALSE.

import io.shiftleft.codepropertygraph.generated.nodes.{AstNode, Call, ControlStructure}
import java.io.PrintWriter

def esc(s: String) = s.replace("\\", "\\\\").replace("\"", "\\\"")
  .replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")

def jsonStrList(xs: Seq[String]) = xs.map(x => "\"" + esc(x) + "\"").mkString(",")

// Every control-structure condition between a node and its method, outermost first.
def ancestorConditions(n: AstNode): List[String] = {
  var conds = List[String]()
  var prev: AstNode = n
  var cur = n.astParent
  while (cur != null && cur.label != "METHOD") {
    if (cur.label == "CONTROL_STRUCTURE") {
      val cs = cur.asInstanceOf[ControlStructure]
      cs.condition.code.headOption.foreach { text =>
        val inCondition = cs.condition.id.l.contains(prev.id)
        val inElse = cs.whenFalse.id.l.contains(prev.id)
        if (inCondition) {
        } else if (inElse) {
          conds = s"!($text)" :: conds
        } else {
          conds = text :: conds
        }
      }
    }
    prev = cur
    cur = cur.astParent
  }
  conds
}

// In a CPG every operator is a CALL, so keeping every CALL line would keep every line
// carrying an expression and R1 would reduce nothing.
val KEEP_LABELS = Set("LOCAL", "CONTROL_STRUCTURE", "RETURN", "JUMP_TARGET")

def isKeptNode(n: AstNode): Boolean = n.label match {
  case "CALL" => !n.asInstanceOf[Call].name.startsWith("<")
  case l      => KEEP_LABELS.contains(l)
}

@main def main(cpgPath: String, outPath: String) = {
  importCpg(cpgPath)

  // Real, named, source-backed methods only: drop Joern's synthetic <global>, <clinit>...
  val methods = cpg.method.filenameNot("<empty>").filterNot(_.name.startsWith("<")).l

  val out = methods.map { m =>
    val base = m.lineNumber.map(_.toInt).getOrElse(0)

    val calls = m.call.filterNot(_.name.startsWith("<")).map { c =>
      val conds = ancestorConditions(c)
      s"""{"name":"${esc(c.name)}","conditions":[${jsonStrList(conds)}]}"""
    }.l

    val idents = m.ast.isIdentifier.name.toSet.toList.sorted

    // Rows relative to the method's first line, so Python indexes candidate.source directly.
    val keep = m.ast
      .filter(isKeptNode)
      .flatMap(_.lineNumber.map(_.toInt))
      .map(_ - base)
      .filter(_ >= 0)
      .toSet.toList.sorted

    s"""{"name":"${esc(m.name)}",""" +
      s""""filename":"${esc(m.filename)}",""" +
      s""""lineNumber":$base,""" +
      s""""code":"${esc(m.code)}",""" +
      s""""calls":[${calls.mkString(",")}],""" +
      s""""identifiers":[${jsonStrList(idents)}],""" +
      s""""keepLines":[${keep.mkString(",")}]}"""
  }

  val writer = new PrintWriter(outPath)
  writer.write("[" + out.mkString(",") + "]")
  writer.close()
  println(s"[dump.sc] wrote ${methods.size} methods to $outPath")
}
