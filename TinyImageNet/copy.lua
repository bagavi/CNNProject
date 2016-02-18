-- copy.lua
--
-- Functions of varying complexity levels to achieve
-- a deep copy in Lua.
--


-- 1. The Problem.
--
-- Here's an example to see why deep copies are useful.
-- Let's say function f receives a table parameter t,
-- and it wants to locally modify that table without
-- affecting the caller. This code fails:
--
-- function f(t)
--  t.a = 3
-- end
--
-- local my_t = {a = 5}
-- f(my_t)
-- print(my_t.a)  --> 3
--
-- This behavior can be hard to work with because, in
-- general, side effects such as input modifications
-- make it more difficult to reason about program
-- behavior.


-- 2. The easy solution.

function copy1(obj)
  if type(obj) ~= 'table' then return obj end
  local res = {}
  for k, v in pairs(obj) do res[copy1(k)] = copy1(v) end
  return res
end

-- This functions works well for simple tables. Since
-- it is a clear, concise function, and since I most
-- often work with simple tables, this is my favorite
-- version.
--
-- There are two aspects this does not handle:
-- * metatables
-- * recursive tables


-- 3. Adding metatable support.

function copy2(obj)
  if type(obj) ~= 'table' then return obj end
  local res = setmetatable({}, getmetatable(obj))
  for k, v in pairs(obj) do res[copy2(k)] = copy2(v) end
  return res
end

-- Well, that wasn't so hard.


-- 4. Supporting recursive structures.
--
-- The issue here is that the following code will
-- get stuck in an infinite loop:
-- 
-- local my_t = {}
-- my_t.a = my_t
-- local t_copy = copy2(my_t)
--
-- This happens to both copy1 and copy2, which each
-- try to make a copy of my_t.a, which involves making
-- a copy of my_t.a.a, which involves making a copy
-- of my_t.a.a.a, etc. The recursive table my_t is
-- perfectly legal, and it's possible to make a
-- deep_copy function that can handle this by tracking
-- which tables it has already started to copy.

function copy3(obj, seen)
  -- Handle non-tables and previously-seen tables.
  if type(obj) ~= 'table' then return obj end
  if seen and seen[obj] then return seen[obj] end

  -- New table; mark it as seen an copy recursively.
  local s = seen or {}
  local res = setmetatable({}, getmetatable(obj))
  s[obj] = res
  for k, v in pairs(obj) do res[copy3(k, s)] = copy3(v, s) end
  return res
end

