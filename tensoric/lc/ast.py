class LCNode:
    pass

class LCVar(LCNode):
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return f"{self.name}"

class LCLam(LCNode):
    def __init__(self, param, body):
        self.param = param
        self.body = body
    def __repr__(self):
        return f"λ{self.param}.({self.body})"

class LCApp(LCNode):
    def __init__(self, fun, arg):
        self.fun = fun
        self.arg = arg
    def __repr__(self):
        return f"({self.fun} {self.arg})"

class LCNum(LCNode):
    def __init__(self, val):
        self.val = val
    def __repr__(self):
        return str(self.val)

class LCSuc(LCNode):
    def __init__(self, val):
        self.val = val
    def __repr__(self):
        return f"+{self.val}"

class LCSwi(LCNode):
    def __init__(self, cond, z_branch, s_branch):
        self.cond = cond
        self.z_branch = z_branch
        self.s_branch = s_branch
    def __repr__(self):
        return f"?{self.cond}{{0:{self.z_branch};+:{self.s_branch};}}"
