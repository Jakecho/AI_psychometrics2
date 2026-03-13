import os
import re

file_path = "/Users/jakecho/Documents/GitHub/AI_psychometrics2/AIG_SME_Toolkit_v2.py"

with open(file_path, "r") as f:
    content = f.read()

# Replacements
content = content.replace("Key Feature Venn Diagram Method", "Cognitive Misconception Mapping (CMM)")
content = content.replace("Key Feature Venn Diagram method", "Cognitive Misconception Mapping (CMM)")
content = content.replace("Key Feature Venn Diagram", "Cognitive Misconception Mapping (CMM)")
content = content.replace("KEY FEATURE VENN DIAGRAM METHOD", "COGNITIVE MISCONCEPTION MAPPING (CMM)")
content = content.replace("Venn diagram in the knowledge space", "CMM in the knowledge space")
content = content.replace("intersecting circles in a Venn diagram", "feature mapping in CMM")
content = content.replace("Venn diagram logic", "CMM logic")
content = content.replace("Venn Diagram Logic", "CMM Logic")
content = content.replace("Venn Diagram Reasoning", "CMM Reasoning")
content = content.replace("Venn diagram reasoning", "CMM reasoning")
content = content.replace("Venn Reasoning", "CMM Reasoning")
content = content.replace("Venn Diagram Analysis", "CMM Analysis")
content = content.replace("venn_reasoning", "cmm_reasoning")

with open(file_path, "w") as f:
    f.write(content)

print("Replaced all Venn diagram terminology with CMM in AIG_SME_Toolkit_v2.py")
