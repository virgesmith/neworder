
# tests

context("neworder")

test_that("dummy test", {
  result = rcpp_hello_world()
  expect_equal(result[[1]][1], "foo")
  expect_equal(result[[1]][2], "bar")
  expect_equal(result[[2]][1], 0)
  expect_equal(result[[2]][2], 1)
})


