# Colloq

## Lecture 1: Stack. How to build/run/test

### GHC, GHCi
- **GHC** (Glasgow Haskell Compiler) — основной компилятор Haskell.
- **GHCi** — интерактивная среда выполнения (REPL) для Haskell.

### Haskell Project Structure
```
my-project/
├── app/          # Исходный код приложения
├── src/          # Исходный код библиотеки
├── test/         # Тесты
├── package.yaml  # Конфигурация проекта (альтернатива .cabal)
├── my-project.cabal # Метаданные и настройки сборки
├── stack.yaml    # Конфигурация Stack
└── Setup.hs      # Скрипт сборки
```

### Stack. Features
- Инструмент для управления проектами, зависимостями и сборки.
- Изоляция версий компилятора и библиотек.
- Поддержка воспроизводимых сборок.

### How Stack Works. Snapshots
- **Снапшоты** — неизменяемые наборы версий пакетов (LTS, Nightly).
- `stack.yaml` указывает, какой снапшот использовать.
- Локальное кеширование пакетов.

### .cabal и .yaml Files
- **.cabal** — декларативное описание пакета (библиотеки, исполняемые файлы, тесты).
- **stack.yaml** — настройки Stack (снапшот, дополнительные пакеты).

### Basic Commands
```bash
stack new my-project    # Создать новый проект
stack build            # Собрать проект
stack exec my-exe      # Запустить исполняемый файл
stack test             # Запустить тесты
stack ghci             # Запустить GHCi
stack hoogle query     # Поиск в документации
```

---

## Lecture 2: Basic Syntax

### Introduction to Haskell
- чистый
- ленивый
- статически типизированный функциональный язык
- иммутабельный

### Basic GHCi Examples
```haskell
ghci> 3 + 4
7
ghci> (+) 3 4
7
ghci> mod 7 3
1
ghci> 7 `mod` 3
1
```

### Function & Operators Definition

##### Function declaration example
```haskell
addMul :: Int -> Int -> Int -> Int
addMul x y z = x + y * z

greet :: String -> String
greet name = "Hello, " ++ name ++ "!"
```

##### Custom operators declaration
```haskell
infixr 2 || -- infixr - порядок расставления скобок, есть еще infixl и infix
(||) :: Bool -> Bool -> Bool

infixr 3 &&
(&&) :: Bool -> Bool -> Bool
```



### Lists and Functions on Lists
```haskell
list = [1, 2, 3]
head list            -- 1
tail list            -- [2,3]
map (*2) list        -- [2,4,6]
```

### let vs where
- **`let`**: Локальные определения в выражениях (можно использовать в GHCi и внутри do-нотации).
  ```
  let <bingings> in <expression>
  ```
- **`where`**: Привязки в конце определения функции (только в определении).

### Условные выражения
- **`if-then-else`**:
    ```
    if <condition> then <expr1> else <expr2>
    ```
- **Охранные выражения (Guards)**:
  ```haskell
  classify x
    | x < 0 = "Negative"
    | otherwise = "Non-negative"
  ```
- **`case`**:
  ```haskell
  case xs of
    [] -> "Empty"
    (y:ys) -> "Non-empty"
  ```

### Higher-Order Functions
Функции, принимающие/возвращающие другие функции:
```haskell
twice f x = f (f x)
```

### Lambdas
Анонимные функции:
```haskell
\x -> x + 1
map (\x -> x * 2) [1,2,3]
```

### Полиморфизм
- **Параметрический**: Работает для любого типа (`id :: a -> a`).
- **Ad-hoc (Перегрузка)**: Разная реализация для разных типов (через классы типов).

### LANGUAGE Pragmas
- `{-# LANGUAGE TupleSections #-}`: Включить частичное применение кортежей.
- `{-# LANGUAGE LambdaCase #-}`: Синтаксис `\case` для анонимных `case`.
- `{-# LANGUAGE ViewPatterns #-}`: Сопоставление с образцом через функции.

### Каррирование (Частичное применение)
Функции принимают аргументы по одному:
```haskell
add x y = x + y
add3 = add 3        -- Частичное применение
```

#### flip
```haskell
flip :: (a -> b -> c) -> b -> a -> c
flip f b a = f a b

show2 :: Int -> Int -> String
show2 x y = show x ++ " and " ++ show y
showSnd, showFst, showFst' :: Int -> String
showSnd  = show2 1
showFst  = flip show2 2
showFst' = (`show2` 2)
```

```haskell
ghci> showSnd 10
"1 and 10"

ghci> showFst 10
"10 and 2"

ghci> showFst' 42
"42 and 2"
```

### Pattern Matching
Сопоставление с образцом в аргументах:
```haskell
factorial 0 = 1
factorial n = n * factorial (n - 1)
```

```haskell
dropWhile :: (a -> Bool) -> [a] -> [a]
dropWhile _      []  = []
dropWhile p l@(x:xs) = if p x then dropWhile p xs else l
```

### List Comprehension
Генерация списков:
```haskell
[x * 2 | x <- [1..5], x > 2]  -- [6,8,10]
```

### Function Application (`$`) и Composition (`.`)
- **`$`**: Правоассоциативное применение (избегает скобок).
  ```haskell
  infixr 0 $
  ($) :: (a -> b) -> a -> b  -- function application
  f $ x = f x  
  ```
- **`.`**: Композиция функций.
  ```haskell
  infixr 9 .
  (.) :: (b -> c) -> (a -> b) -> (a -> c)  -- same as (b -> c) -> (a -> b) -> a -> c
  f . g = \x -> f (g x)
  ```

#### Functions simplifying
```haskell
stringsTransform :: [String] -> [String]

stringsTransform l = map (\s -> map toUpper s) (filter (\s -> length s == 5) l)
Copy

stringsTransform l = map (\s -> map toUpper s) $ filter (\s -> length s == 5) l

stringsTransform l = map (map toUpper) $ filter ((== 5) . length) l

stringsTransform = map (map toUpper) . filter ((== 5) . length)
```

### Lazy Evaluation
Выражения вычисляются только когда нужны:
- **Решето Эратосфена**: Бесконечный список простых чисел.
- **Числа Фибоначчи**: `fibs = 0 : 1 : zipWith (+) fibs (tail fibs)`.
- **Repmin**: Алгоритм, одновременно вычисляющий минимум и заменяющий все элементы на него.

![lazy eval](https://i.ibb.co/dw8Vkp3Y/image.png)

### Normal form

##### NF
NF = все подвыражения вычислены
```haskell
42

(2, "hello")

\x -> (x + 1)
```

##### No NF
```haskell
1 + 2

(\x -> x + 1) 2

"he" ++ "llo"

(1 + 1, 2 + 2)
```

##### Weak head normal form
внешняя функция должна быть конструктором или лямбдой
```haskell
10

2 : [1, 3]

'h' : ("e" ++ "llo")

[1, 2 * 2, 3 ^ 3]

[4, length undefined]

Just (2 + 7)

(1 + 2, 3 + 4)

\x -> x + 10

\zx -> foldr (+) zx [1, 2, 3]
```

##### No WNF
```haskell
1 + 1

1 + (2 + 3)

(\x -> x + 10) 3

length [1, 2, 3]

Just (2 + 7) >>= 
    \n -> Just $ n * 2
```

#### ! PATTERN MATCHING REGUIRES WNF
---

## Lecture 3: Datas, Classes, Instances

### type (Синонимы типов)
```haskell
type String = [Char]
```

### ADT (Algebraic Data Types)
- **Sum Types** (Варианты):
  ```haskell
  data Bool = False | True
  ```
- **Product Types** (Записи):
  ```haskell
  data Point = Point Int Int
  ```

```haskell
      ┌─ type name
      │
      │       ┌─ constructor name (or constructor tag)
      │       │
data User = MkUser Int String String
 │                  │    │      │
 │                  └────┴──────┴── types of fields
 │
 └ "data" keyword
```

##### Parametrized
```haskell
data Point2D a = Point2D a a  -- constructor name can be the same as type name

maxCoord :: Point2D Int -> Int
maxCoord (Point2D x y) = max x y
```

##### Recursive
```haskell
data List a = Nil | Cons a (List a)
``` 

##### Record syntax
```haskell
data User = User 
    { uid      :: Int
    , login    :: String
    , password :: String 
    }

-- init
ivan :: User
ivan = User { login    = "Ivan"
            , password = "123" 
            , uid      = 1
            }

-- update
cloneIvan :: User
cloneIvan = ivan { uid = 2 }  -- User 2 "Ivan" "123"

-- pattern matching
isIvan :: User -> Bool
isIvan User{ login = userName } = userName == "Ivan"
```

### Расширения для записей
- `-XDuplicateRecordFields `: Разрешить поля с одинаковыми именами в разных типах.
- `{-# LANGUAGE RecordWildCards #-}`: можно относиться к полям как к переменным, используя синтаксис `Person{..}`.
  ```haskell
  {-# LANGUAGE RecordWildCards #-}

    data User = User 
        { uid      :: Int
        , login    :: String
        , password :: String 
        } deriving (Show)

    toUnsafeString :: User -> String
    toUnsafeString User{ uid = 0, .. } = "ROOT: " ++ login ++ ", " ++ password
    toUnsafeString User{..}            = login ++ ":" ++ password
  ```

### newtype
Если тип имеет только один конструктор с только одним параметром, то вместо `data` его можно определить через `newtype`. Это более эффективно в runtime.
```haskell
newtype Meters = Meters Double
```

### Type Classes
- **`class`**: Объявление класса. Работает примерно как интерфейсы в других языках.
```haskell
class Printable p where
    printMe :: p -> String
```
- **`instance`**: Реализация интерфейса.
```haskell
instance Printable Foo where
    printMe Foo = "Foo"
    printMe Bar = "Bar (whatever)"
```

### Примеры стандартных классов
- `Eq` - равенство
```haskell
class Eq a where  
    (==) :: a -> a -> Bool  
    (/=) :: a -> a -> Bool
 
    x == y = not (x /= y)  
    x /= y = not (x == y)
    {-# MINIMAL (==) | (/=) #-}  -- minimal complete definition
```
- `Ord` - сравнение
```haskell
data Ordering = LT | EQ | GT
class Eq a => Ord a where -- simplified definition
   compare              :: a -> a -> Ordering
   (<), (<=), (>=), (>) :: a -> a -> Bool

   compare x y
        | x == y    =  EQ
        | x <= y    =  LT
        | otherwise =  GT

   x <= y           =  compare x y /= GT
   x <  y           =  compare x y == LT
   x >= y           =  compare x y /= LT
   x >  y           =  compare x y == GT

```
- `Num` - числа
```haskell
class Num a where
    {-# MINIMAL (+), (*), abs, signum, fromInteger, (negate | (-)) #-}

    (+), (-), (*)       :: a -> a -> a  -- self-explained
    negate              :: a -> a       -- unary negation
    abs                 :: a -> a       -- absolute value
    signum              :: a -> a       -- sign of number, abs x * signum x == x
    fromInteger         :: Integer -> a -- used for numeral literals polymorphism

    x - y               = x + negate y
    negate x            = 0 - x
```
- `Show` - строковое представление
```haskell
-- simplified version; used for converting things into String
class Show a where
    show :: a -> String
```
- `Read` - чтение из строки
```haskell
-- simplified version; used for parsing thigs from String
class Read a where
    read :: String -> a
```

### Deriving
Автоматическая генерация экземпляров:
```haskell
data TrafficLight = Red | Yellow | Green | Blue
    deriving (Eq, Ord, Enum, Bounded, Show, Read, Ix)
```

### Расширения для deriving
- `GeneralizedNewtypeDeriving`: Наследование экземпляров для `newtype`.

### Modules Cheatsheet
```haskell
module Lib 
       ( module Exports
       , FooB1 (..), FooB3 (FF)
       , Data.List.nub, C.isUpper
       , fooA, bazA, BAZB.isLower
       ) where

import           Foo.A
import           Foo.B     (FooB2 (MkB1), 
                            FooB3 (..))
import           Prelude   hiding (print)
import           Bar.A     (print, (<||>))
import           Bar.B     ()

import           Baz.A     as BAZA  
import qualified Data.List
import qualified Data.Char as C hiding (chr)
import qualified Baz.B     as BAZB (isLower)

import qualified Foo.X     as Exports
import qualified Foo.Y     as Exports
```

```haskell
module Foo.B 
       ( FooB1, FooB2 (..), 
         FooB3 (FF, val)
       ) where

data FooB1 = MkFooB1
data FooB2 = MkB1 | MkB2
data FooB3 = FF { val :: Int }
```

```haskell
module Bar.B () where

class Printable p where 
    printMe :: p -> String

instance Printable Int where 
    printMe = show
```

### Church-encoding ADT
Представление алгебраических типов через функции (например, `Maybe a` как `forall r. (a -> r) -> r -> r`).

---

## Lecture 4: Basic Typeclasses

### Semigroup и Monoid
- **Semigroup**
```haskell
class Semigroup m where
    (<>) :: m -> m -> m

-- Associativity law for Semigroup: 
--   1. (x <> y) <> z ≡ x <> (y <> z)
```

- **Monoid**
```haskell
class Semigroup m => Monoid m where
    mempty :: m

-- Identity laws for Monoid:
--   1. (x <> y) <> z ≡ x <> (y <> z)
--   2. x <> mempty ≡ x
--   3. mempty <> x ≡ x
```

#### Examples
```haskell
instance Semigroup [a] where
    (<>) = (++)

newtype Sum     a = Sum     { getSum     :: a }
newtype Product a = Product { getProduct :: a }
newtype Max   a = Max   { getMax   :: a    }  -- max
newtype Min   a = Min   { getMin   :: a    }  -- min
newtype Any     = Any   { getAny   :: Bool }  -- ||
newtype All     = All   { getAll   :: Bool }  -- &&
newtype First a = First { getFirst :: a }     -- first value
newtype Last  a = Last  { getLast  :: a }     -- last value

instance Num a => Semigroup (Sum a) where
  Sum x <> Sum y = Sum (x + y)
instance Num a => Semigroup (Product a) where
  Product x <> Product y = Product (x * y)
```

```haskell
class Semigroup a => Monoid a where
    mempty  :: a
    mappend :: a -> a -> a -- (<>) by default
    mconcat :: [a] -> a

-- моноид для списков
instance Monoid [a] where
    mempty = []

-- моноид для пар
instance (Monoid a, Monoid b) => Monoid (a, b) where
                         mempty = (         mempty,          mempty)
    (a1, b1) `mappend` (a2, b2) = (a1 `mappend` a2, b1 `mappend` b2)

instance Num a => Monoid (Sum a) where
    mempty = Sum 0

instance Num a => Monoid (Product a) where
    mempty = Product 1
```

### foldr и foldl
```haskell
foldr :: (a -> b -> b) -> b -> [a] -> b
foldl :: (b -> a -> b) -> b -> [a] -> b

-- Examples:

foldr (+) 0 [1,2,3] 
= 1 + (foldr (+) 0 [2,3])
= 1 + (2 + (foldr (+) 0 [3]))
= 1 + (2 + (3 + (foldr (+) 0 [])))
= 1 + (2 + (3 + 0))
= 6

foldl (+) 0 [1,2,3]
= foldl (+) (0+1) [2,3]
= foldl (+) ((0+1)+2) [3]
= foldl (+) (((0+1)+2)+3) []
= ((0+1)+2)+3
= 6
```

<a href="https://ibb.co/4wyG1Snz"><img src="https://i.ibb.co/twftL2Tg/image.png" alt="image" border="0"></a>

### Foldable
```haskell
class Foldable t where
    {-# MI  NIMAL foldMap | foldr #-}

    -- Here `t m` is type-container. For example: `Maybe m`.
    -- Fold transforms container `t m` into single `m`
    -- using step-by-step application of m's monoid function.
    fold    :: Monoid m => t m -> m
    foldMap :: Monoid m => (a -> m) -> t a -> m

    -- like reduce
    -- takes  as arguments: mapper, init-val and container
    foldr   :: (a -> b -> b) -> b -> t a -> b


-- instances:
instance Foldable [] where
    foldr :: (a -> b -> b) -> b -> [a] -> b
    foldr _ z []     =  z
    foldr f z (x:xs) =  x `f` foldr f z xs

instance Foldable Maybe where
    foldr :: (a -> b -> b) -> b -> Maybe a -> b
    foldr _ z Nothing  = z
    foldr f z (Just x) = f x z
```

### Functor
```haskell
class Functor f where -- f :: Type -> Type
    fmap :: (a -> b) -> f a -> f b

-- laws:
-- 1. Identity 
--    fmap id      ≡ id
-- 2. Composition
--    fmap (f . g) ≡ fmap f . fmap g
```

##### Examples:
```haskell
instance Functor Maybe where
    fmap :: (a -> b) -> Maybe a -> Maybe b
    fmap f (Just x) = Just (f x)
    fmap _ Nothing  = Nothing
```

`<$>` - аналог `$` для функторов:
```haskell
infixl 4 <$>
(<$>) :: Functor f => (a -> b) -> f a -> f b
(<$>) = fmap

firstPostTitle = getPostTitle <$> findPost 1
```

**Arrow functor:**
```haskell
-- стрелочный оператор: (->) Int Bool ~ (Int -> Bool)

ghci> :kind (->)
(->) :: Type -> Type -> Type  -- this kind signature is enough for us
```

```haskell
-- Частичное применение: (->) r = \a -> r -> a

instance Functor ((->) r) where
    -- using mapper (a -> b) to get from func (r -> a) new func (r -> b)
    -- (r -> a) == ((->) r) a - this is why it's `Functor ((->) r)`
    fmap :: (a -> b) -> (r -> a) -> (r -> b)
    fmap = (.)  -- fmap f = \g -> f . g
```

### Applicative
Все вычисления производятся "внутри" контейнеров (или контекста). Внутри контейнеров находятся как функции, так и аргументы

```haskell
class Functor f => Applicative f where  -- f :: Type -> Type
    pure  :: a -> f a
    -- arg1: func inside container,
    -- arg2: val inside container,
    -- returns (func val) inside container
    (<*>) :: f (a -> b) -> f a -> f b
    liftA2 :: (a -> b -> c) -> f a -> f b -> f c  -- since GHC 8.2.1
```

##### Examples:
```haskell
instance Applicative Maybe where
    pure :: a -> Maybe a
    pure = Just
    
    (<*>) :: Maybe (a -> b) -> Maybe a -> Maybe b
    Nothing <*> _         = Nothing
    Just f  <*> something = fmap f something

instance Applicative [] where
    pure :: a -> [a]
    pure x    = [x]

    (<*>) :: [a -> b] -> [a] -> [b]
    fs <*> xs = [f x | f <- fs, x <- xs]

ghci> [(*2), (+3)] <*> [1, 2, 3]
[2, 4, 6, 4, 5, 6]
```

##### Arrow-applicative
```haskell
instance Applicative ((->) r) where
    pure :: a -> r -> a  -- the K combinator!
    pure x  = \_ -> x -- constant lambda

    -- arg1 mapper (a -> b) with context (context is (->) r)
    -- arg2 val with context
    -- res is combination in context
    (<*>) :: (r -> a -> b) -> (r -> a) -> r -> b  -- the S combinator!
    -- x is `r`, (f x) has type (a -> b), (g x) has type b
    f <*> g = \x -> f x (g x)
```

##### Applicative laws
```
1. Identity
   pure id <*> v ≡ v

2. Composition
   pure (.) <*> u <*> v <*> w ≡ u <*> (v <*> w)

3. Homomorphism
   pure f <*> pure x ≡ pure (f x)

4. Interchange
   u <*> pure y ≡ pure ($ y) <*> u
   
5. Compatibility w/ Functors
   fmap f u ≡ pure f <*> u
```

##### Applicative-style
```haskell
data User = User
    { userFirstName :: String
    , userLastName  :: String
    , userEmail     :: String
    }
type Profile = [(String, String)]

profileExample = 
    [ ("first_name", "Pat"            )
    , ("last_name" , "Brisbin"        )
    , ("email"     , "me@pbrisbin.com")
    ]

lookup "first_name" p :: Maybe String

buildUser :: Profile -> Maybe User
buildUser p = User
    <$> lookup "first_name" p -- поднимаем функцию-конструктор User в контекст `Maybe`
    <*> lookup "last_name"  p -- продолжаем каррировать функцию User внутри контекста
    <*> lookup "email"      p -- применяем 3ий и последний аргумент к User внутри контекста Maybe

-- point-free version
buildUser :: Profile -> Maybe User
buildUser = liftA3 (liftA3 User) -- то же самое но непонятными словами ;)
                   (lookup "first_name")
                   (lookup "last_name")
                   (lookup "email")
```

### liftAN
```haskell
liftA2 :: (a -> b -> c) -> f a -> f b -> f c  -- meaning: we pass a BINARY function into context & apply inside context

-- f <$> x - вносит f в контекст
-- (f <$> x) <*> y - применяет внутри контекста
liftA2 f x y = f <$> x <*> y
```

`LiftAN` - то же, но для N аргументов.

### Alternative
**Alternative** - класс типов для аппликативных функторов с возможностью выбора и обработки неудач. Имеет:
- `empty` — представление "неудачи" или "нейтрального элемента"
- `(<|>)` — оператор выбора (попробовать первую альтернативу, если неудача — вторую)

```haskell
class Applicative f => Alternative f where
    empty :: f a
    (<|>) :: f a -> f a -> f a
```

##### Examples:
```haskell
instance Alternative Maybe where
    empty :: Maybe a
    empty = Nothing

    (<|>) :: Maybe a -> Maybe a -> Maybe a
    Nothing <|> r = r
    l       <|> _ = l

instance Alternative [] where
    empty :: [a]
    empty = []
    
    (<|>) :: [a] -> [a] -> [a]
    (<|>) = (++)
```

### List Comprehension Syntax Sugar
`[x*y | x <- [1,2], y <- [3,4]]` преобразуется в `liftA2 (*) [1,2] [3,4]`.

### Traversable
**Traversable** - класс типов для структур данных, которые можно "пройти", применяя эффект к каждому элементу и собирая результаты.

```haskell
class (Functor t, Foldable t) => Traversable t where
    -- arg1 mapper (a -> f b) that returns res inside context `f`
    -- arg2 val (t a) - a in container
    -- returns res inside context `f` with container `t` structure preserved
    traverse  :: Applicative f => (a -> f b) -> t a -> f (t b)
    sequenceA :: Applicative f => t (f a) -> f (t a)
```

##### Examples:
```haskell
instance Traversable Maybe where
    traverse :: Applicative f => (a -> f b) -> Maybe a -> f (Maybe b)
    traverse _ Nothing  = pure Nothing
    traverse f (Just x) = Just <$> f x

instance Traversable [] where
    traverse :: Applicative f => (a -> f b) -> [a] -> f [b]
    traverse f = foldr consF (pure [])
      where 
        consF x ys = (:) <$> f x <*> ys
```

### Automatic Deriving
- `DeriveFunctor`, `DeriveFoldable`, `DeriveTraversable`: Автовывод экземпляров.

### Types scheme
![types](https://i.ibb.co/jvxChgVg/image.png)

### Phantom Types
Параметры типа, не используемые в значении:
```haskell
newtype Hash a = MkHash String -- `a` is a phantom type, 
                               -- not present in any of the constructors
```

##### Use case
```haskell
class Hashable a where
    hash :: a -> Hash a

hashPair :: Int -> Hash (Int, Int)
hashPair n = hash (n, n)

hashPair :: Int -> Hash (Int, Int)
hashPair n = hash n  -- no longer a valid definition!
```

### Type Extensions
- `ScopedTypeVariables`: Лексическая область переменных типа.
- `TypeApplications`: Явная передача аргументов типа (`id @Int 5`).
- `AllowAmbiguousTypes`: Разрешить неоднозначные типы (требует `TypeApplications`).

##### Scoping
```haskell
id :: a -> a
-- this is the same definitions
id :: forall a . a -> a

prepend2 :: a -> [a] -> [a]
prepend2 x xs = pair ++ xs 
  where 
    pair :: [a] -- it's different `a`, compiler will replace it with a1
    pair = [x, x]

-- there it is shown better
prepend2 :: forall a . a -> [a] -> [a]
prepend2 x xs = pair ++ xs 
  where 
    pair :: forall a . [a]
    pair = [x, x] 
```

---

## Lecture 5: Monads

### Typed Holes
- `_` слева от равенства - если при паттерн-матчинге подойдет любой вариант
- `_` справа от равенства - `typed hole`. Такой код никогда не скомпилируется, но ошибка компилятора подскажет ожидаемый тип

Также `_` можно использовать в описании типа аналогично `typed hole`:
```haskell
poid :: Int -> _
poid = show
```
```haskell
Lecture5.hs:78:16: error:
    • Found type wildcard ‘_’ standing for ‘String’
      To use the inferred type, enable PartialTypeSignatures
    • In the type signature: poid :: Int -> _
   |
78 | poid :: Int -> _
   |                
```

### What is a Monad?
**Variable** - container for data.
**Monad** - container for sequentially composable computation.
Monad is a general way to describe idea of computations where you can combine computations in a such way so that next computation depends on result of previous computation.

- Типкласс:
  ```haskell
  class Applicative m => Monad m where
    (>>=) :: m a -> (a -> m b) -> m b
  ```

### Monad Laws
1. **Left identity**: `return a >>= f ≡ f a`
2. **Right identity**: `m >>= return ≡ m`
3. **Associativity**: `(m >>= f) >>= g ≡ m >>= (\x -> f x >>= g)`

### State Monad
```haskell
newtype State s a = State { runState :: s -> (a, s) }
```
- Для управления изменяемым состоянием.

### Reader Monad
```haskell
newtype Reader r a = Reader { runReader :: r -> a }
```
- Для доступа к конфигурации.

### Maybe Monad
- Обработка отсутствующих значений (null-safety).

### Either Monad
- Обработка ошибок с информацией.

---

## Lecture 5 2.0 Monads

### Parser Combinators
- **Parser Type**: `newtype Parser a = Parser (String -> [(a, String)])`
- **Basic Parsers**: `char`, `string`, `many`, `some`.
- **Instances**: `Functor`, `Applicative`, `Monad`, `Alternative`.

### Property-Based Testing
- **hspec**: Структура для модульных тестов.
- **hedgehog**: Генерация данных и свойства.
- **tasty**: Комбинирование тестов.

### Continuations
- **Cont**: `newtype Cont r a = Cont { runCont :: (a -> r) -> r }`
- Монада для управления потоком выполнения.

---

## Lecture 6: RealWorld

### IO Monad
- Монада для взаимодействия с внешним миром.

### do-notation
Синтаксический сахар для цепочек монадических вычислений:
```haskell
main = do
  name <- getLine
  putStrLn $ "Hello, " ++ name
```

### Расширения
- `ApplicativeDo`: Оптимизация для аппликативных вычислений.
- `RebindableSyntax`: Использовать пользовательские `>>=`, `return`.

### Lazy I/O
- Чтение данных по требованию (может вызывать проблемы с ресурсами).

### FFI (Foreign Function Interface)
- Вызов функций из C и других языков.

### Mutable Data
- **IORef**: Изменяемые ссылки.
- **IOArray**: Изменяемые массивы.

### Exceptions
- `catch`, `throwIO`, `bracket` (безопасное приобретение/освобождение ресурсов).

### unsafePerformIO
- Небезопасное выполнение IO в чистом коде (только для экспертов).

### Efficient String Representations
- **Text**: Unicode строки.
- **ByteString**: Бинарные данные.

---

## Lecture 7: Monad Transformers

### Monads as Effects
- Каждая монада добавляет свой эффект (State, Reader, Exception).

### Composing Monads
- **Трансформеры** позволяют комбинировать монады.

### MonadIO
- Класс для поднятия IO в другие монады:
  ```haskell
  class MonadIO m where
    liftIO :: IO a -> m a
  ```

### MonadTrans
- Класс для трансформеров:
  ```haskell
  class MonadTrans t where
    lift :: Monad m => m a -> t m a
  ```

### MaybeT Transformer
```haskell
newtype MaybeT m a = MaybeT { runMaybeT :: m (Maybe a) }
```

### ReaderT Transformer
```haskell
newtype ReaderT r m a = ReaderT { runReaderT :: r -> m a }
```

### Сравнение Transformers vs Old Types
- Трансформеры композируемы, "старые" монады (как `State`) — нет.

### MonadThrow / MonadError
- Абстракции для обработки ошибок.

### mtl Style
- Использование классов типов (как `MonadReader`) для избежания явных `lift`.
```haskell
foo :: MonadReader Config m => m String
```





---
# ! это уже не коллок !
# Lecture 7: Monad Transformers

## Monads as Effects

**Монады как эффекты** — это подход, где каждая монада представляет определённый тип побочного эффекта:

- **`Maybe`** — эффект возможного отсутствия значения
- **`Either e`** — эффект ошибки с информацией
- **`State s`** — эффект изменяемого состояния
- **`Reader r`** — эффект чтения из окружения
- **`Writer w`** — эффект записи лога
- **`IO`** — эффект ввода-вывода

```haskell
-- Чистая функция
pureAdd :: Int -> Int -> Int
pureAdd x y = x + y

-- Функция с эффектом Maybe (возможное отсутствие значения)
safeDivide :: Int -> Int -> Maybe Int
safeDivide _ 0 = Nothing
safeDivide x y = Just (x `div` y)

-- Функция с эффектом State (изменяемое состояние)
incrementCounter :: State Int Int
incrementCounter = do
    count <- get
    put (count + 1)
    return count
```

## Composing Monads

**Проблема композиции монад**: монады не композируются напрямую. Тип `m (n a)` не является монадой автоматически.

```haskell
-- НЕ РАБОТАЕТ так просто:
-- getLine :: IO String
-- readMaybe :: String -> Maybe Int
-- getLine >>= readMaybe  -- ОШИБКА ТИПОВ: IO String -> (String -> Maybe Int) -> ???

-- Решение: монадные трансформеры
```

## MonadIO

**`MonadIO`** — класс типов для поднятия `IO` действий в любую другую монаду, поддерживающую `IO`.

```haskell
class Monad m => MonadIO m where
    liftIO :: IO a -> m a

-- Примеры использования
readFileIO :: MonadIO m => FilePath -> m String
readFileIO path = liftIO (readFile path)

printIO :: (MonadIO m, Show a) => a -> m ()
printIO x = liftIO (print x)

-- Использование в стеке трансформеров
example :: ReaderT Config IO ()
example = do
    config <- ask
    content <- liftIO $ readFile (configFile config)  -- Поднимаем IO в ReaderT Config IO
    liftIO $ putStrLn content
```

## MonadTrans Type Class

**`MonadTrans`** — базовый класс для всех монадных трансформеров.

```haskell
class MonadTrans t where
    lift :: Monad m => m a -> t m a

-- Законы:
-- 1. lift . return = return
-- 2. lift (m >>= f) = lift m >>= (lift . f)
```

## MaybeT Transformer

**`MaybeT`** — трансформер, добавляющий эффект возможного неуспеха.

```haskell
newtype MaybeT m a = MaybeT { runMaybeT :: m (Maybe a) }

instance MonadTrans MaybeT where
    lift m = MaybeT (fmap Just m)

-- Пример: комбинация Maybe и IO
readConfigFile :: MaybeT IO String
readConfigFile = do
    content <- liftIO $ readFile "config.json"
    if null content
        then MaybeT (return Nothing)  -- Неуспех
        else return content           -- Успех

-- Запуск: runMaybeT readConfigFile :: IO (Maybe String)
```

## ReaderT Transformer

**`ReaderT`** — самый популярный трансформер, добавляющий эффект read-only окружения.

```haskell
newtype ReaderT r m a = ReaderT { runReaderT :: r -> m a }

instance MonadTrans (ReaderT r) where
    lift m = ReaderT (\_ -> m)

-- Пример использования
type Config = (String, Int)

getDatabaseConnection :: ReaderT Config IO String
getDatabaseConnection = do
    (host, port) <- ask
    liftIO $ putStrLn $ "Connecting to " ++ host ++ ":" ++ show port
    return "connection-handle"

-- Запуск: runReaderT getDatabaseConnection ("localhost", 5432) :: IO String
```

## Comparison of Transformers and Old Types

### Старый подход (конкретные монады)
```haskell
-- Каждая комбинация эффектов требует новой монады
data AppState = AppState { counter :: Int, log :: [String] }

-- Собственная монада для State + Writer
newtype MyMonad a = MyMonad { runMyMonad :: AppState -> (a, AppState, [String]) }

-- Проблема: для добавления Maybe эффекта нужно переписать всё
```

### Трансформерный подход
```haskell
-- Композиция через трансформеры
type App = ReaderT Config (StateT AppState (WriterT [String] IO))

-- Легко добавлять/убирать эффекты
type AppWithError = ReaderT Config (StateT AppState (ExceptT String (WriterT [String] IO)))
```

**Преимущества трансформеров:**
- Композируемость
- Переиспользование кода
- Гибкость в построении стека
- Стандартные реализации

## MonadThrow Type Class

**`MonadThrow`** — для монад, которые могут бросать исключения.

```haskell
class Monad m => MonadThrow m where
    throwM :: Exception e => e -> m a

-- Пример использования
import Control.Exception (Exception, throwIO)
import Control.Monad.Catch

data ConfigError = FileNotFound | InvalidFormat
    deriving (Show, Exception)

loadConfig :: (MonadThrow m, MonadIO m) => FilePath -> m String
loadConfig path = do
    exists <- liftIO $ doesFileExist path
    if exists
        then liftIO $ readFile path
        else throwM FileNotFound
```

## MonadError Type Class

**`MonadError`** — более мощная версия `MonadThrow` с возможностью перехвата ошибок.

```haskell
class MonadThrow m => MonadError e m | m -> e where
    throwError :: e -> m a
    catchError :: m a -> (e -> m a) -> m a

-- Пример использования
type AppError = String

safeDivide :: (MonadError AppError m, MonadIO m) => Int -> Int -> m Int
safeDivide x y = do
    liftIO $ putStrLn $ "Dividing " ++ show x ++ " by " ++ show y
    if y == 0
        then throwError "Division by zero"
        else return (x `div` y)

-- Обработка ошибок
calculator :: (MonadError AppError m, MonadIO m) => m Int
calculator = do
    result <- safeDivide 10 0 `catchError` \err -> do
        liftIO $ putStrLn $ "Error: " ++ err
        return 0  -- Значение по умолчанию
    return result
```

## mtl Style of Transformation

**mtl (Monad Transformer Library) стиль** — использование классов типов вместо конкретных трансформеров для повышения гибкости кода.

```haskell
-- Вместо конкретного стека:
-- complicatedFunction :: ReaderT Config (StateT AppState IO) Result

-- Используем классы типов:
complicatedFunction :: (MonadReader Config m, MonadState AppState m, MonadIO m) => m Result
complicatedFunction = do
    config <- ask
    state <- get
    liftIO $ putStrLn "Processing..."
    -- Логика функции
    put (state { counter = counter state + 1 })
    return "result"

-- Теперь эту функцию можно использовать с ЛЮБЫМ стеком,
-- который предоставляет необходимые эффекты
```

### Полный пример mtl стиля:

```haskell
{-# LANGUAGE FlexibleContexts #-}

import Control.Monad.Reader
import Control.Monad.State
import Control.Monad.IO.Class

-- Конфигурация и состояние
data Config = Config { dbHost :: String, dbPort :: Int }
data AppState = AppState { connectionCount :: Int, lastError :: Maybe String }

-- Функция в mtl стиле
connectToDB :: (MonadReader Config m, MonadState AppState m, MonadIO m) => m String
connectToDB = do
    config <- ask
    state <- get
    
    liftIO $ putStrLn $ 
        "Connecting to " ++ dbHost config ++ ":" ++ show (dbPort config)
    
    -- Обновляем состояние
    put $ state { connectionCount = connectionCount state + 1 }
    
    -- Имитация подключения
    return "db-connection-handle"

-- Та же функция работает с разными стеками:

-- Стек 1: Reader + State + IO
type App1 = ReaderT Config (StateT AppState IO)
runApp1 :: App1 a -> Config -> AppState -> IO (a, AppState)
runApp1 app config state = runStateT (runReaderT app config) state

-- Стек 2: Reader + IO (без State)
type App2 = ReaderT Config IO
runApp2 :: App2 a -> Config -> IO a
runApp2 app config = runReaderT app config

-- Для App2 нужно адаптировать состояние, но сама логика функции не меняется!
```

### Преимущества mtl стиля:
- **Полиморфизм**: код работает с любым подходящим стеком
- **Тестируемость**: легко подменить реализацию для тестов
- **Гибкость**: можно менять стек без переписывания бизнес-логики
- **Композиция**: функции легко комбинируются

### Замечание:
В вопросе упоминались `MaybeIO` и `CoroutineT` — это либо опечатки, либо нестандартные трансформеры. Вероятно, имелись в виду:
- `MaybeT` (трансформер для Maybe)
- `ContT` (трансформер для продолжений) или другие специализированные трансформеры