;; Boolean logic components mapping Church Booleans natively

(def true (lambda (t f) t))
(def false (lambda (t f) f))

(def not (lambda (b) (b false true)))
(def and (lambda (a b) (a b false)))
(def or (lambda (a b) (a true b)))
(def xor (lambda (a b) (a (not b) b)))

;; Generic IF statement with strict laziness enforcement.
;; Interaction Calculus reduces everywhere. To prevent infinite recursion,
;; we must block the recursive call structurally. We pass `id` into the closure
;; and apply it to the recursive function, effectively making it an unknown Application
;; until the branch is selected!
(def id (lambda (x) x))
(def if (lambda (cond t_lazy f_lazy) ((cond t_lazy f_lazy) id)))

;; Helper for Interaction-Calculus native primitive numbers (0, +N) to return Church Booleans
(def is_zero (lambda (n) (match-num n true (lambda (pred) false))))
(def pred (lambda (n) (match-num n 0 (lambda (p) p))))

;; -------------------------------------------
;; Z-Combinator for Recursion
;; -------------------------------------------
(def Z (lambda (f) ((lambda (x) (f (lambda (v) ((x x) v)))) (lambda (x) (f (lambda (v) ((x x) v)))))))

;; -------------------------------------------
;; While Loop
;; Iterates state until condition is true
;; while_loop(cond_func, body_func, state)
;; -------------------------------------------
(def while_rec (lambda (while cond_func body_func state)
  (if (cond_func state)
      (lambda (d) (((d while) cond_func) body_func (body_func state)))
      (lambda (d) state))))
(def while (Z while_rec))

;; -------------------------------------------
;; For Loop
;; Applies a function N times to an initial state
;; for_loop(n, body_func, state)
;; -------------------------------------------
(def for_rec (lambda (for n body_func state)
  (if (is_zero n)
      (lambda (d) state)
      (lambda (d) (((d for) (pred n)) body_func (body_func state))))))
(def for_loop (Z for_rec))

;; Examples
(and true false)
(xor true true)
(xor false true)

;; Run a 'for' loop 3 times, adding 2 each time to state 0: 0 -> 2 -> 4 -> 6
(def add2 (lambda (x) (suc (suc x))))
(for_loop (suc (suc (suc 0))) add2 0)
