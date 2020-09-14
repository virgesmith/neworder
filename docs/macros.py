# macros for mkdocs-macros-plugin

import os

_inline_code_styles = {
  ".py": "python",
  ".sh": "bash",
  ".h": "cpp",
  ".cpp": "cpp",
  ".c": "c",
  ".rs": "rs",
  ".js": "js"
}

def define_env(env):

  # @env.macro
  # def test(s):
  #   return "```some python code here: %s```\n" % s

  @env.macro
  def include_snippet(filename, tag=None):
    """ looks for code in <filename> between lines containing "#!<tag>" """
    full_filename = os.path.join(env.project_dir, filename)

    _, file_type = os.path.splitext(filename)
    # default to # for comment string, and text for inline code style
    code_style = _inline_code_styles.get(file_type, "text")

    with open(full_filename, 'r') as f:
      lines = f.readlines()

    if tag:
      tag = "!%s!" % tag
      span = []
      for i,l in enumerate(lines):
        if tag in l:
          span.append(i)
      if len(span) != 2:
        return "```ERROR %s (%s) too few/many tags (%s) for '%s'```" % (filename, code_style, len(span), tag)
      lines = lines[span[0]+1: span[1]]

    #line_range = lines[start_line+1:end_line]
    return "```%s\n" % code_style + "".join(lines) + "```" 

# if __name__ == "__main__":
#   print(_include_snippet("examples/chapter1/model.py", "tag"))