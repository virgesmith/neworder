
import importlib

module_name = "neworder"
md = "docs/api.md"

type_mapping = {
  "<class 'pybind11_builtins.pybind11_type'>": "class",
  "<class 'instancemethod'>": "instance method",
  "<class 'builtin_function_or_method'>": "function",
  "<class 'module'>": "module"
}

def format_overloads(lines):
  for i, l in enumerate(lines):
    if l[:2] == "1." or l[:2] == "2." or l[:2] == "3.":
      lines[i] = "```python\n" + l[2:] + "\n```"
  return lines

def format_heading(l, a, t):
  typestring = '!!! note "%s"\n\n' % t
  return "%s `%s`\n\n%s" % ("#"*l, ".".join(a), typestring)

def format_docstr(m, t):
  if not m.__doc__:
    return "__doc__ empty\n"
  doc = m.__doc__
  lines = format_overloads(doc.split("\n"))
  for i,l in enumerate(lines):
    lines[i] = l.lstrip()
  if t in ["instance method", "function"]:
    lines[0] = "```python\n" + lines[0] + "\n```"
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
    f.write(format_heading(l, parents + [sm.__name__], t))
    f.write(format_docstr(sm, t))
    if "class" in t or "module" in t:
      recurse_attrs(sm, parents + [sm.__name__], l+1, f)
  parents = parents[:-2]

module = importlib.import_module(module_name)

with open(md, "w") as f:
  f.write("# API Reference\n")
  recurse_attrs(module, ["neworder"], 2, f)

