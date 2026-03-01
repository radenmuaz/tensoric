import numpy as np

# 32-bit Term representation equivalents
# Tags
VAR = 0x00
LAM = 0x01
APP = 0x02
ERA = 0x03
NUM = 0x04
SUC = 0x05
SWI = 0x06
TMP = 0x07

SP0 = 0x08
SP1 = 0x09
SP2 = 0x0A
SP3 = 0x0B
SP4 = 0x0C
SP5 = 0x0D
SP6 = 0x0E
SP7 = 0x0F

DX0 = 0x10
DX1 = 0x11
DX2 = 0x12
DX3 = 0x13
DX4 = 0x14
DX5 = 0x15
DX6 = 0x16
DX7 = 0x17

DY0 = 0x18
DY1 = 0x19
DY2 = 0x1A
DY3 = 0x1B
DY4 = 0x1C
DY5 = 0x1D
DY6 = 0x1E
DY7 = 0x1F

# Masks (32-bit equivalent, though we might use separated arrays)
TERM_SUB_MASK = 0x80000000
TERM_TAG_MASK = 0x7C000000
TERM_VAL_MASK = 0x03FFFFFF

MAX_NODES = 33554432

class StaticIC:
    def __init__(self, size=MAX_NODES):
        # We simulate the heap using a single int32 numpy array to mimic uint32 packed terms
        # but using python's unbounded ints or explicitly np.uint32 for bitwise ops
        self.heap = np.zeros(size, dtype=np.uint32)
        self.stack = np.zeros(size, dtype=np.uint32)
        self.heap_pos = 0
        self.stack_pos = 0
        self.interactions = 0

    def alloc(self, n):
        ptr = self.heap_pos
        self.heap_pos += n
        return ptr
        
    def make_term(self, is_sub, tag, val):
        term = 0
        if is_sub:
            term |= TERM_SUB_MASK
        term |= (tag << 26)
        term |= (val & TERM_VAL_MASK)
        return term
        
    def get_tag(self, term):
        return (term & TERM_TAG_MASK) >> 26
        
    def get_val(self, term):
        return term & TERM_VAL_MASK
        
    def is_sub(self, term):
        return (term & TERM_SUB_MASK) != 0

    def clear_sub(self, term):
        return term & (~TERM_SUB_MASK & 0xFFFFFFFF)
        
    def make_sub(self, term):
        return term | TERM_SUB_MASK
        
    # Helpers
    def ic_make_sup(self, lab, val):
        return self.make_term(False, SP0 + lab, val)
        
    def ic_make_co0(self, lab, val):
        return self.make_term(False, DX0 + lab, val)
        
    def ic_make_co1(self, lab, val):
        return self.make_term(False, DY0 + lab, val)
        
    def ic_make_era(self):
        return self.make_term(False, ERA, 0)
        
    def ic_make_num(self, val):
        return self.make_term(False, NUM, val)
        
    def ic_make_suc(self, val):
        return self.make_term(False, SUC, val)
        
    def ic_make_swi(self, val):
        return self.make_term(False, SWI, val)
        
    # Constructors
    def ic_lam(self, bod):
        loc = self.alloc(1)
        self.heap[loc] = bod
        return loc
        
    def ic_app(self, fun, arg):
        loc = self.alloc(2)
        self.heap[loc] = fun
        self.heap[loc+1] = arg
        return loc
        
    def ic_sup(self, lft, rgt):
        loc = self.alloc(2)
        self.heap[loc] = lft
        self.heap[loc+1] = rgt
        return loc
        
    def ic_dup(self, val):
        loc = self.alloc(1)
        self.heap[loc] = val
        return loc
        
    def ic_suc(self, num):
        loc = self.alloc(1)
        self.heap[loc] = num
        return loc
        
    def ic_swi(self, num, ifz, ifs):
        loc = self.alloc(3)
        self.heap[loc] = num
        self.heap[loc+1] = ifz
        self.heap[loc+2] = ifs
        return loc

    # Interaction tracking
        self.interactions = 0

    def ic_app_lam(self, app, lam):
        self.interactions += 1
        app_loc = self.get_val(app)
        lam_loc = self.get_val(lam)
        arg = self.heap[app_loc + 1]
        bod = self.heap[lam_loc + 0]
        self.heap[lam_loc] = self.make_sub(arg)
        return bod

    def ic_app_sup(self, app, sup):
        self.interactions += 1
        app_loc = self.get_val(app)
        sup_loc = self.get_val(sup)
        # The true 32-bit C code does: TERM_LAB(sup) = (TERM_TAG(term) & LAB_MAX) where LAB_MAX is 7
        # And SUP base tag is SP0
        sup_tag = self.get_tag(sup)
        sup_lab = sup_tag - SP0

        arg = self.heap[app_loc + 1]
        lft = self.heap[sup_loc + 0]
        rgt = self.heap[sup_loc + 1]

        dup_loc = self.alloc(1)
        app1_loc = self.alloc(2)

        self.heap[dup_loc] = arg
        x0 = self.ic_make_co0(sup_lab, dup_loc)
        x1 = self.ic_make_co1(sup_lab, dup_loc)

        self.heap[sup_loc + 1] = x0
        
        self.heap[app1_loc + 0] = rgt
        self.heap[app1_loc + 1] = x1

        self.heap[app_loc + 0] = self.make_term(False, APP, sup_loc)
        self.heap[app_loc + 1] = self.make_term(False, APP, app1_loc)

        return self.ic_make_sup(sup_lab, app_loc)
        
    def ic_app_era(self, app, era):
        self.interactions += 1
        return era

    def ic_dup_era(self, dup, era):
        self.interactions += 1
        dup_loc = self.get_val(dup)
        era_term = self.ic_make_era()
        self.heap[dup_loc] = self.make_sub(era_term)
        return era_term

    def ic_dup_lam(self, dup, lam):
        self.interactions += 1
        dup_loc = self.get_val(dup)
        lam_loc = self.get_val(lam)
        dup_tag = self.get_tag(dup)
        is_co0 = (dup_tag >= DX0 and dup_tag <= DX7)
        dup_lab = dup_tag - DX0 if is_co0 else dup_tag - DY0

        bod = self.heap[lam_loc + 0]

        alloc_start = self.alloc(5)
        lam0_loc = alloc_start
        lam1_loc = alloc_start + 1
        sup_loc = alloc_start + 2
        dup_new_loc = alloc_start + 4

        self.heap[sup_loc + 0] = self.make_term(False, VAR, lam0_loc)
        self.heap[sup_loc + 1] = self.make_term(False, VAR, lam1_loc)

        self.heap[lam_loc] = self.make_sub(self.ic_make_sup(dup_lab, sup_loc))
        self.heap[dup_new_loc] = bod

        self.heap[lam0_loc] = self.ic_make_co0(dup_lab, dup_new_loc)
        self.heap[lam1_loc] = self.ic_make_co1(dup_lab, dup_new_loc)

        if is_co0:
            self.heap[dup_loc] = self.make_sub(self.make_term(False, LAM, lam1_loc))
            return self.make_term(False, LAM, lam0_loc)
        else:
            self.heap[dup_loc] = self.make_sub(self.make_term(False, LAM, lam0_loc))
            return self.make_term(False, LAM, lam1_loc)
            
    def ic_dup_sup(self, dup, sup):
        self.interactions += 1
        dup_loc = self.get_val(dup)
        sup_loc = self.get_val(sup)
        
        dup_tag = self.get_tag(dup)
        sup_tag = self.get_tag(sup)
        
        is_co0 = (dup_tag >= DX0 and dup_tag <= DX7)
        dup_lab = dup_tag - DX0 if is_co0 else dup_tag - DY0
        sup_lab = sup_tag - SP0

        lft = self.heap[sup_loc + 0]
        rgt = self.heap[sup_loc + 1]

        if dup_lab == sup_lab:
            if is_co0:
                self.heap[dup_loc] = self.make_sub(rgt)
                return lft
            else:
                self.heap[dup_loc] = self.make_sub(lft)
                return rgt
        else:
            sup_start = self.alloc(4)
            sup0_loc = sup_start
            sup1_loc = sup_start + 2

            dup_lft_loc = sup_loc + 0
            dup_rgt_loc = sup_loc + 1

            self.heap[sup0_loc + 0] = self.ic_make_co0(dup_lab, dup_lft_loc)
            self.heap[sup0_loc + 1] = self.ic_make_co0(dup_lab, dup_rgt_loc)
            
            self.heap[sup1_loc + 0] = self.ic_make_co1(dup_lab, dup_lft_loc)
            self.heap[sup1_loc + 1] = self.ic_make_co1(dup_lab, dup_rgt_loc)

            self.heap[dup_lft_loc] = lft
            self.heap[dup_rgt_loc] = rgt

            if is_co0:
                self.heap[dup_loc] = self.make_sub(self.ic_make_sup(sup_lab, sup1_loc))
                return self.ic_make_sup(sup_lab, sup0_loc)
            else:
                self.heap[dup_loc] = self.make_sub(self.ic_make_sup(sup_lab, sup0_loc))
                return self.ic_make_sup(sup_lab, sup1_loc)
                
    def ic_suc_num(self, suc, num):
        self.interactions += 1
        num_val = self.get_val(num)
        return self.ic_make_num(num_val + 1)
        
    def ic_suc_era(self, suc, era):
        self.interactions += 1
        return era
        
    def ic_suc_sup(self, suc, sup):
        self.interactions += 1
        sup_loc = self.get_val(sup)
        sup_tag = self.get_tag(sup)
        sup_lab = sup_tag - SP0

        lft = self.heap[sup_loc + 0]
        rgt = self.heap[sup_loc + 1]

        suc0_loc = self.ic_suc(lft)
        suc1_loc = self.ic_suc(rgt)

        res_loc = self.alloc(2)
        self.heap[res_loc + 0] = self.ic_make_suc(suc0_loc)
        self.heap[res_loc + 1] = self.ic_make_suc(suc1_loc)

        return self.ic_make_sup(sup_lab, res_loc)
        
    def ic_swi_num(self, swi, num):
        self.interactions += 1
        swi_loc = self.get_val(swi)
        num_val = self.get_val(num)

        ifz = self.heap[swi_loc + 1]
        ifs = self.heap[swi_loc + 2]

        if num_val == 0:
            return ifz
        else:
            app_loc = self.alloc(2)
            self.heap[app_loc + 0] = ifs
            self.heap[app_loc + 1] = self.ic_make_num(num_val - 1)
            return self.make_term(False, APP, app_loc)
            
    def ic_swi_era(self, swi, era):
        self.interactions += 1
        return era
        
    def ic_swi_sup(self, swi, sup):
        self.interactions += 1
        swi_loc = self.get_val(swi)
        sup_loc = self.get_val(sup)
        sup_tag = self.get_tag(sup)
        sup_lab = sup_tag - SP0

        lft = self.heap[sup_loc + 0]
        rgt = self.heap[sup_loc + 1]
        ifz = self.heap[swi_loc + 1]
        ifs = self.heap[swi_loc + 2]

        dup_z_loc = self.alloc(1)
        dup_s_loc = self.alloc(1)

        self.heap[dup_z_loc] = ifz
        self.heap[dup_s_loc] = ifs

        z0 = self.ic_make_co0(sup_lab, dup_z_loc)
        z1 = self.ic_make_co1(sup_lab, dup_z_loc)
        s0 = self.ic_make_co0(sup_lab, dup_s_loc)
        s1 = self.ic_make_co1(sup_lab, dup_s_loc)

        swi0_loc = self.ic_swi(lft, z0, s0)
        swi1_loc = self.ic_swi(rgt, z1, s1)

        res_loc = self.alloc(2)
        self.heap[res_loc + 0] = self.make_term(False, SWI, swi0_loc)
        self.heap[res_loc + 1] = self.make_term(False, SWI, swi1_loc)

        return self.ic_make_sup(sup_lab, res_loc)

    def ic_dup_num(self, dup, num):
        self.interactions += 1
        dup_loc = self.get_val(dup)
        self.heap[dup_loc] = self.make_sub(num)
        return num

    def is_dup(self, tag):
        return tag >= DX0 and tag <= DY7

    def is_sup(self, tag):
        return tag >= SP0 and tag <= SP7

    # WHNF Evaluator Port
    def ic_whnf(self, term):
        stop = self.stack_pos
        next_term = term
        
        while True:
            tag = self.get_tag(next_term)
            
            if tag == VAR:
                val_loc = self.get_val(next_term)
                val = self.heap[val_loc]
                if self.is_sub(val):
                    next_term = self.clear_sub(val)
                    continue
            elif self.is_dup(tag):
                val_loc = self.get_val(next_term)
                val = self.heap[val_loc]
                if self.is_sub(val):
                    next_term = self.clear_sub(val)
                    continue
                else:
                    self.stack[self.stack_pos] = next_term
                    self.stack_pos += 1
                    next_term = val
                    continue
            elif tag == APP:
                val_loc = self.get_val(next_term)
                self.stack[self.stack_pos] = next_term
                self.stack_pos += 1
                next_term = self.heap[val_loc]
                continue
            elif tag == SUC:
                val_loc = self.get_val(next_term)
                self.stack[self.stack_pos] = next_term
                self.stack_pos += 1
                next_term = self.heap[val_loc]
                continue
            elif tag == SWI:
                val_loc = self.get_val(next_term)
                self.stack[self.stack_pos] = next_term
                self.stack_pos += 1
                next_term = self.heap[val_loc]
                continue
                
            if self.stack_pos == stop:
                return next_term
                
            self.stack_pos -= 1
            prev = self.stack[self.stack_pos]
            ptag = self.get_tag(prev)
            
            if ptag == APP:
                if tag == LAM:
                    next_term = self.ic_app_lam(prev, next_term)
                    continue
                elif self.is_sup(tag):
                    next_term = self.ic_app_sup(prev, next_term)
                    continue
                elif tag == ERA:
                    next_term = self.ic_app_era(prev, next_term)
                    continue
            elif self.is_dup(ptag):
                if tag == LAM:
                    next_term = self.ic_dup_lam(prev, next_term)
                    continue
                elif self.is_sup(tag):
                    next_term = self.ic_dup_sup(prev, next_term)
                    continue
                elif tag == ERA:
                    next_term = self.ic_dup_era(prev, next_term)
                    continue
                elif tag == NUM:
                    next_term = self.ic_dup_num(prev, next_term)
                    continue
            elif ptag == SUC:
                if tag == NUM:
                    next_term = self.ic_suc_num(prev, next_term)
                    continue
                elif self.is_sup(tag):
                    next_term = self.ic_suc_sup(prev, next_term)
                    continue
                elif tag == ERA:
                    next_term = self.ic_suc_era(prev, next_term)
                    continue
            elif ptag == SWI:
                if tag == NUM:
                    next_term = self.ic_swi_num(prev, next_term)
                    continue
                elif self.is_sup(tag):
                    next_term = self.ic_swi_sup(prev, next_term)
                    continue
                elif tag == ERA:
                    next_term = self.ic_swi_era(prev, next_term)
                    continue
                    
            self.stack[self.stack_pos] = prev
            self.stack_pos += 1
            
            if self.stack_pos == stop:
                return next_term
                
            while self.stack_pos > stop:
                self.stack_pos -= 1
                prev = self.stack[self.stack_pos]
                ptag = self.get_tag(prev)
                val_loc = self.get_val(prev)
                if ptag == APP or ptag == SWI or self.is_dup(ptag):
                    self.heap[val_loc] = next_term
                next_term = prev
                
            return next_term

    def ic_normal(self, term):
        term = self.ic_whnf(term)
        tag = self.get_tag(term)
        loc = self.get_val(term)
        
        if tag == ERA or tag == NUM:
            return term
        elif tag == LAM:
            self.heap[loc] = self.ic_normal(self.heap[loc])
            return term
        elif tag == APP:
            self.heap[loc+0] = self.ic_normal(self.heap[loc+0])
            self.heap[loc+1] = self.ic_normal(self.heap[loc+1])
            return term
        elif self.is_sup(tag):
            self.heap[loc+0] = self.ic_normal(self.heap[loc+0])
            self.heap[loc+1] = self.ic_normal(self.heap[loc+1])
            return term
        elif tag == SUC:
            self.heap[loc] = self.ic_normal(self.heap[loc])
            return term
        elif tag == SWI:
            self.heap[loc+0] = self.ic_normal(self.heap[loc+0])
            self.heap[loc+1] = self.ic_normal(self.heap[loc+1])
            self.heap[loc+2] = self.ic_normal(self.heap[loc+2])
            return term
        return term
