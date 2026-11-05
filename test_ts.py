from tree_sitter_language_pack import get_parser

code = b"def add(a, b):\n    return a + b\n"
tree = get_parser("python").parse(code)
print(tree.root_node.type)
print([c.type for c in tree.root_node.children])