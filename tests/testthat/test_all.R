
# tests

context("neworder")

test_that("urand01 test", {
  u01 = urand01_vector(10000);
  expect_gte(min(u01), 0.0)
  expect_lte(max(u01), 1.0)
})

test_that("age test", {
  
  df = data.frame(Age=sample(1:85, 1000, replace=T), Gender=sample(1:2, 1000, replace=T))
  age_mean = mean(df$Age)
  sex_mean = mean(df$Gender)
  
  age_column(df, "Age")
  expect_equal(mean(df$Age), age_mean + 1)
  expect_equal(mean(df$Gender), sex_mean)
})


test_that("select test", {
  
  df = data.frame(Age=sample(1:85, 1000, replace=T), Gender=sample(1:2, 1000, replace=T))
  
  # all females have babies, 100% fertility rate
  selected = neworder::select_at_random(df, "Gender", c(0, 0.0, 1.0))

  expect_equal(sum(selected), nrow(df[df$Gender==2,]))

  # all age 30 have babies even if male
  p = rep(0, 86)
  p[31] = 1
  selected = neworder::select_at_random(df, "Age", p)
  
  expect_equal(sum(selected), nrow(df[df$Age==30,]))
})

