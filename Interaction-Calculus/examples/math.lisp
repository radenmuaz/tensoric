;; Simple math library defining Addition and Multiplication over IC

;; -------------------------------------------
;; Church Booleans and Generic IF
;; -------------------------------------------
(def true (lambda (t f) t))
(def false (lambda (t f) f))

;; Generic IF statement.
(def id (lambda (x) x))
(def if (lambda (cond t_lazy f_lazy) ((cond t_lazy f_lazy) id)))

;; Helpers mapping native IC primitives to Booleans and decrementors
(def is_zero (lambda (n) (match-num n true (lambda (pred) false))))
(def pred (lambda (n) (match-num n 0 (lambda (p) p))))

;; -------------------------------------------
;; Z-Combinator and Arithmetic
;; -------------------------------------------
(def Z (lambda (f) ((lambda (x) (f (lambda (v) ((x x) v)))) (lambda (x) (f (lambda (v) ((x x) v)))))))

(def add_rec (lambda (add m n) 
  (if (is_zero m)
      (lambda (d) n)
      (lambda (d) (suc (((d add) (pred m)) n))))))
(def add (Z add_rec))

(def mul_rec (lambda (mul m n)
  (if (is_zero m)
      (lambda (d) 0)
      (lambda (d) (add n (((d mul) (pred m)) n))))))
(def mul (Z mul_rec))

;; 1. Recursive Factorial
(def fact_rec (lambda (fact n) 
  (if (is_zero n)
      (lambda (d) (suc 0))
      (lambda (d) (mul n ((d fact) (pred n)))))))
(def fact (Z fact_rec))

;; 2. Iterative Factorial using a tail-call accumulator
(def fact_iter_rec (lambda (fact_iter n acc)
  (if (is_zero n)
      (lambda (d) acc)
      (lambda (d) (((d fact_iter) (pred n)) (mul n acc))))))
(def fact_iter (lambda (n) ((Z fact_iter_rec) n (suc 0))))

;; 3. Recursive Fibonacci
(def fib_rec (lambda (fib n)
  (if (is_zero n)
      (lambda (d) 0)
      (lambda (d)
        (if (is_zero (pred n))
            (lambda (d2) (suc 0))
            (lambda (d2) (add ((d2 fib) (pred n)) ((d2 fib) (pred (pred n))))))))))
(def fib (Z fib_rec))

;; 4. Iterative Fibonacci
(def fib_iter_rec (lambda (fib_iter n a b)
  (if (is_zero n)
      (lambda (d) a)
      (lambda (d) ((((d fib_iter) (pred n)) b) (add a b))))))
(def fib_iter (lambda (n) ((Z fib_iter_rec) n 0 (suc 0))))

;; Evaluation assertions
(fact (suc (suc (suc (suc 0)))))
(fact_iter (suc (suc (suc (suc 0)))))

(fib (suc (suc (suc (suc (suc (suc 0)))))))
(fib_iter (suc (suc (suc (suc (suc (suc 0)))))))
