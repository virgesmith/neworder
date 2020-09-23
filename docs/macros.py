# macros for mkdocs-macros-plugin

import os
import requests

_inline_code_styles = {
  ".py": "python",
  ".sh": "bash",
  ".h": "cpp",
  ".cpp": "cpp",
  ".c": "c",
  ".rs": "rs",
  ".js": "js",
  ".md": None
}

def define_env(env):

  # @env.macro
  # def test(s):
  #   return "```some python code here: %s```\n" % s

  @env.macro
  def insert_version():
    with open("./VERSION") as f:
      return f.readline().rstrip()

  @env.macro
  def insert_doi():
    response = requests.get('https://zenodo.org/api/records', params={'q': '4031821'})

    if response.status_code == 200:
      result = response.json()
      if "hits" in result and \
         "hits" in result["hits"] and \
         len(result["hits"]["hits"]) > 0 and \
         "doi" in result["hits"]["hits"][0]:
        return result["hits"]["hits"][0]["doi"]
      else:
        return "[json error retrieving doi]"
    return "[http error %d retrieving doi]" % response.status_code


  @env.macro
  def include_snippet(filename, tag=None, show_filename=True):
    """ looks for code in <filename> between lines containing "!<tag>!" """
    full_filename = os.path.join(env.project_dir, filename)

    _, file_type = os.path.splitext(filename)
    # default to literal "text" for inline code style
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

    if show_filename:
      footer = "\n[file: **%s**]\n" % filename
    else:
      footer = ""
    #line_range = lines[start_line+1:end_line]
    if code_style is not None:
      return "```%s\n" % code_style + "".join(lines) + "```" + footer
    else:
      return "".join(lines) + footer

# if __name__ == "__main__":
#   print(_include_snippet("examples/chapter1/model.py", "tag"))