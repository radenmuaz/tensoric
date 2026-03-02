from staticic import *

DP0_TYPE = 100
DP1_TYPE = 101

class VarNameTable:
    def __init__(self):
        self.locations = []
        self.types = []
        self.names = []

    def index_to_var_name(self, index):
        if index < 26:
            return chr(ord('a') + index)
        else:
            first = (index - 26) // 26
            second = (index - 26) % 26
            return chr(ord('a') + first) + chr(ord('a') + second)

    def add_variable(self, location, type_val):
        basic_type = type_val
        if type_val >= DX0 and type_val <= DX7:
            basic_type = DP0_TYPE
        elif type_val >= DY0 and type_val <= DY7:
            basic_type = DP1_TYPE

        for i in range(len(self.locations)):
            if self.locations[i] == location and self.types[i] == basic_type:
                return self.names[i]

        count = len(self.names)
        self.locations.append(location)
        self.types.append(basic_type)

        if basic_type == VAR:
            name = self.index_to_var_name(count)
        elif basic_type == DP0_TYPE:
            name = f"a{count}"
        elif basic_type == DP1_TYPE:
            name = f"b{count}"
        else:
            name = f"?{count}"

        self.names.append(name)
        return name

    def get_var_name(self, location, type_val):
        basic_type = type_val
        if type_val >= DX0 and type_val <= DX7:
            basic_type = DP0_TYPE
        elif type_val >= DY0 and type_val <= DY7:
            basic_type = DP1_TYPE

        for i in range(len(self.locations)):
            if self.locations[i] == location and self.types[i] == basic_type:
                return self.names[i]
        return "?"

class DupTable:
    def __init__(self):
        self.locations = []
        self.labels = []

    def register(self, location, label):
        for i in range(len(self.locations)):
            if self.locations[i] == location:
                if self.labels[i] != label:
                    raise Exception("Label mismatch for duplication")
                return False
        self.locations.append(location)
        self.labels.append(label)
        return True

class Stringifier:
    def __init__(self, ic):
        self.ic = ic
        self.var_table = VarNameTable()
        self.dup_table = DupTable()
        self.output = ""

    def assign_var_ids(self, term):
        tag = self.ic.get_tag(term)
        val = self.ic.get_val(term)
        
        if tag == VAR:
            subst = self.ic.heap[val]
            if self.ic.is_sub(subst):
                self.assign_var_ids(self.ic.clear_sub(subst))
        elif self.ic.is_dup(tag):
            lab = tag - DX0 if (tag >= DX0 and tag <= DX7) else tag - DY0
            subst = self.ic.heap[val]
            if self.ic.is_sub(subst):
                self.assign_var_ids(self.ic.clear_sub(subst))
            else:
                if self.dup_table.register(val, lab):
                    self.assign_var_ids(subst)
        elif tag == LAM:
            self.var_table.add_variable(val, VAR)
            self.assign_var_ids(self.ic.heap[val])
        elif tag == APP:
            self.assign_var_ids(self.ic.heap[val])
            self.assign_var_ids(self.ic.heap[val + 1])
        elif tag == ERA or tag == NUM:
            pass
        elif self.ic.is_sup(tag):
            self.assign_var_ids(self.ic.heap[val])
            self.assign_var_ids(self.ic.heap[val + 1])
        elif tag == SUC:
            self.assign_var_ids(self.ic.heap[val])
        elif tag == SWI:
            self.assign_var_ids(self.ic.heap[val])
            self.assign_var_ids(self.ic.heap[val+1])
            self.assign_var_ids(self.ic.heap[val+2])

    def stringify_duplications(self):
        for i in range(len(self.dup_table.locations)):
            dup_loc = self.dup_table.locations[i]
            self.var_table.add_variable(dup_loc, DP0_TYPE)
            self.var_table.add_variable(dup_loc, DP1_TYPE)

        for i in range(len(self.dup_table.locations)):
            dup_loc = self.dup_table.locations[i]
            lab = self.dup_table.labels[i]
            val_term = self.ic.heap[dup_loc]

            var0 = self.var_table.get_var_name(dup_loc, DP0_TYPE)
            var1 = self.var_table.get_var_name(dup_loc, DP1_TYPE)

            self.output += f"! &{lab}{{{var0},{var1}}} = "
            self.stringify_term(val_term)
            self.output += ";\n"

    def stringify_term(self, term):
        tag = self.ic.get_tag(term)
        val = self.ic.get_val(term)

        if tag == VAR:
            subst = self.ic.heap[val]
            if self.ic.is_sub(subst):
                self.stringify_term(self.ic.clear_sub(subst))
            else:
                name = self.var_table.get_var_name(val, VAR)
                self.output += name
        elif self.ic.is_dup(tag):
            co_type = DP0_TYPE if (tag >= DX0 and tag <= DX7) else DP1_TYPE
            subst = self.ic.heap[val]
            if self.ic.is_sub(subst):
                self.stringify_term(self.ic.clear_sub(subst))
            else:
                name = self.var_table.get_var_name(val, co_type)
                self.output += name
        elif tag == LAM:
            var_name = self.var_table.get_var_name(val, VAR)
            self.output += f"λ{var_name}."
            self.stringify_term(self.ic.heap[val])
        elif tag == APP:
            self.output += "("
            self.stringify_term(self.ic.heap[val])
            self.output += " "
            self.stringify_term(self.ic.heap[val+1])
            self.output += ")"
        elif tag == ERA:
            self.output += "*"
        elif self.ic.is_sup(tag):
            lab = tag - SP0
            self.output += f"&{lab}{{"
            self.stringify_term(self.ic.heap[val])
            self.output += ","
            self.stringify_term(self.ic.heap[val+1])
            self.output += "}"
        elif tag == NUM:
            self.output += str(val)
        elif tag == SUC:
            self.output += "+"
            self.stringify_term(self.ic.heap[val])
        elif tag == SWI:
            self.output += "?"
            self.stringify_term(self.ic.heap[val])
            self.output += "{0:"
            self.stringify_term(self.ic.heap[val+1])
            self.output += ";+:"
            self.stringify_term(self.ic.heap[val+2])
            self.output += ";}"
        else:
            self.output += "<?unknown term>"

    def ic_print(self, term):
        self.assign_var_ids(term)
        self.stringify_duplications()
        self.stringify_term(term)
        return self.output

def print_term(ic, term):
    s = Stringifier(ic)
    return s.ic_print(term)
