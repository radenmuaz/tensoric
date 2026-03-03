class ICNode:
    pass

class ICVar(ICNode):
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return f"{self.name}"

class ICLam(ICNode):
    def __init__(self, param, body):
        self.param = param
        self.body = body
    def __repr__(self):
        return f"λ{self.param}.({self.body})"

class ICApp(ICNode):
    def __init__(self, fun, arg):
        self.fun = fun
        self.arg = arg
    def __repr__(self):
        return f"({self.fun} {self.arg})"

class ICDup(ICNode):
    def __init__(self, var0, var1, val, body, lab=0):
        self.var0 = var0
        self.var1 = var1
        self.val = val
        self.body = body
        self.lab = lab
    def __repr__(self):
        return f"! &{self.lab}{{{self.var0},{self.var1}}} = {self.val};\n{self.body}"

class ICRep(ICNode):
    def __init__(self, var0, var1, val, body, level=0, delta_l=0, delta_r=0):
        self.var0 = var0
        self.var1 = var1
        self.val = val
        self.body = body
        self.level = level
        self.delta_l = delta_l
        self.delta_r = delta_r
    def __repr__(self):
        return f"REP[L{self.level} dL{self.delta_l} dR{self.delta_r}]{{{self.var0},{self.var1}}} = {self.val};\n{self.body}"

class ICEra(ICNode):
    def __repr__(self):
        return "*"

class ICNum(ICNode):
    def __init__(self, val):
        self.val = val
    def __repr__(self):
        return str(self.val)

class ICSuc(ICNode):
    def __init__(self, val):
        self.val = val
    def __repr__(self):
        return f"+{self.val}"

class ICSwi(ICNode):
    def __init__(self, cond, z_branch, s_branch):
        self.cond = cond
        self.z_branch = z_branch
        self.s_branch = s_branch
    def __repr__(self):
        return f"?{self.cond}{{0:{self.z_branch};+:{self.s_branch};}}"
