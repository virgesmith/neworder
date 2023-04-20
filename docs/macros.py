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

  @env.macro
  def insert_zenodo_field(*keys: str):
    """ This is the *released* version not the dev one """
    try:
      response = requests.get('https://zenodo.org/api/deposit/depositions/7838395', params={'access_token': os.getenv("ZENODO_PAT")})
      response.raise_for_status()
      result = response.json()
      for k in keys:
        result = result[k]
      return result

    except Exception as e:
      return f"{e.__class__.__name__}:{e} while retrieving {keys}"


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
      tag = f"!{tag}!"
      span = []
      for i, l in enumerate(lines):
        if tag in l:
          span.append(i)
      if len(span) != 2:
        return f"```ERROR {filename} ({code_style}) too few/many tags ({len(span)}) for '{tag}'```"
      lines = lines[span[0] + 1: span[1]]

    if show_filename:
      footer = f"\n[file: **{filename}**]\n"
    else:
      footer = ""
    if code_style is not None:
      return f"```{code_style}\n{''.join(lines)}```{footer}"
    else:
      return "".join(lines) + footer

