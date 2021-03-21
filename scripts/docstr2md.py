
import importlib

module_name = "neworder"
md = "docs/api.md"

type_mapping = {
  "<class 'pybind11_builtins.pybind11_type'>": "class",
  "<class 'instancemethod'>": "instance method",
  "<class 'builtin_function_or_method'>": "function",
  "<class 'module'>": "module"
}

def badge(t):
  colour = {
    "class": "darkgreen",
    "instance method": "orange",
    "function": "red",
    "module": "blue"
  }
  h = ""
  return "![%s](https://img.shields.io/badge/%s-%s-%s)" % (t, h, t, colour[t])


def format_overloads(lines):
  for i, l in enumerate(lines):
    if l[:2] == "1." or l[:2] == "2." or l[:2] == "3." or l[:2] == "4.":
      lines[i] = "```python\n" + l[2:].replace("_neworder_core", "neworder") + "\n```"
  return lines

def format_heading(l, a, t):
  return "%s %s `%s`\n\n" % ("#"*l, badge(t), ".".join(a))

def format_docstr(m, t):
  if not m.__doc__:
    return "__doc__ empty\n"
  doc = m.__doc__
  lines = format_overloads(doc.split("\n"))
  for i,l in enumerate(lines):
    lines[i] = l.lstrip()
  if t in ["instance method", "function"]:
    lines[0] = "```python\n" + lines[0].replace("_neworder_core", "neworder") + "\n```"
  return "\n".join(lines) + "\n"


def recurse_attrs(m, parents, l, f):
  attrs = [a for a in dir(m) if a[:2] != "__" or a == "__init__"]
  #print("parents=%s" % ".".join(parents))
  for a in attrs:
    sm = getattr(m, a)
    #print(str(type(sm)))
    t = type_mapping.get(str(type(sm)), None)
    #t = str(type(sm))
    if t is None: continue
    if t != "instance method" and t != "function" or (t == "function" and l == 2):
      f.write("---\n\n")
    # if t == "module":
    #   l = 1
    name = sm.__name__.replace("_neworder_core", "neworder")
    f.write(format_heading(l, [name], t))
    f.write(format_docstr(sm, t))
    if "class" in t or "module" in t:
      recurse_attrs(sm, parents + [name], l+1, f)
  parents = parents[:-2]

module = importlib.import_module(module_name)

with open(md, "w") as f:
  f.write("# ![module](https://img.shields.io/badge/-module-blue) `neworder`\n")
  recurse_attrs(module, ["neworder"], 2, f)

