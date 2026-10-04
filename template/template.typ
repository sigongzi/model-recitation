#let font = (
  // 1. 主文本：英文用 JetBrains Mono（或 Fira Code），中文回退到思源宋体
  main: ("Times New Roman", "Source Han Serif"),
  // 2. 代码块：使用专门的代码字体，带圆角且字符宽度一致
  mono: ("Cascadia Mono", "Source Han Serif"),
  cjk: "Source Han Serif",
)

#let docstyle(
  body
) = {
  // 设置正文字体
  set text(font: font.main, lang: "zh")

  show link: underline

  // 行内代码和代码块使用 mono 字体
  show raw: set text(font: font.mono)

  body
}