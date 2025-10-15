from utils.common import clean_latex_format

s1 = "\\left( 3, \\frac{\\pi}{2} \\right)"
s2 = "##Step 1: Calculate the radial coordinate $r$ $r = \\sqrt{0^2 + 3^2} = \\sqrt{9} = 3$, ##Step 2: Calculate the angular coordinate $\\theta$ $\\theta = \\tan^{-1}\\left(\\frac{3}{0}\\right) = \\frac{\\pi}{2}$, final_answer: $\\boxed{\\left(3, \\frac{\\pi}{2}\\right)}$"
s1 = clean_latex_format(s1)
s2 = clean_latex_format(s2)
print("s1:",s1)
print("s2:",s2)
print(s1 in s2)