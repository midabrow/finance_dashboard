

# ============================================================
# 🔹 COLORING TERMINAL OUTPUT
# ============================================================
class clr:
    bold = '\033[1m'             
    orange = '\033[38;5;208m'  
    blue = '\033[38;5;75m'    
    reset = '\033[0m' 

    @staticmethod
    def print_colored_text():
        print(clr.orange + "This is dark orange text!" + clr.reset)
        print(clr.blue + "This is light blue text!" + clr.reset)
        print(clr.bold + "This is bold text!" + clr.reset)
        print(clr.bold + clr.orange + "This is bold dark orange text!" + clr.reset)