check_values <- function(problem){
  if (problem == 1){

    raven_crt_plot_test = ggplot(test_df, aes(x=ravens.score_test, y=cognitive_reflection_survey.correct_proportion_test,)) +
        geom_jitter(width=.2, height=.02)+ 
        geom_smooth(method='lm', se=FALSE)+
        labs(x="Raven's score", y="CRT score")
    assert_that(all.equal(raven_crt_plot$data, raven_crt_plot_test$data))
  } else if (problem ==2) {
    lm_result_test = lm(cognitive_reflection_survey.correct_proportion_test ~ ravens.score_test, data=test_df)
    assert_that(all.equal(coef(lm_result), coef(lm_result_test)))
  } else if (problem ==3){
    r_hat_test = coef(lm_result)[2] * sd(test_df$ravens.score_test)/sd(test_df$cognitive_reflection_survey.correct_proportion_test)
    assert_that(r_hat_test == r_hat)
  } else if (problem == 4){
    ravens_crt_corr_test = cor.test(test_df$ravens.score_test,
        test_df$cognitive_reflection_survey.correct_proportion_test)
    assert_that(ravens_crt_corr_test$estimate == ravens_crt_corr$estimate)
  } else if (problem ==5){
    ravens_retest_plot_test = ggplot(retest_df, aes(x=ravens.score_test, y=ravens.score_retest)) +
      geom_jitter(width=.2, height=.2)+
      geom_smooth(method='lm', se=FALSE) +
      geom_abline(intercept=0, slope=1, color='red')+
      labs(x="Raven's score on test", y="Raven's score on re-test")
    assert_that(all.equal(ravens_retest_plot_test$data, ravens_retest_plot$data))
  } else if (problem == 6){
    ravens_df_test = retest_df %>%
      dplyr::select(ravens.score_test, ravens.score_retest)
    assert_that(all.equal(ravens_df_test, ravens_df))
    icc_ravens_test = irr::icc(ravens_df, model="oneway", type="consistency", unit="single")
    assert_that(icc_ravens$value == icc_ravens_test$value)
  } else if (problem ==7){
    crt_df_test = retest_df %>%
      select(cognitive_reflection_survey.correct_proportion_test, cognitive_reflection_survey.correct_proportion_retest)
      icc_crt_test = irr::icc(crt_df, model="twoway", type="consistency", unit="single")
      assert_that(all.equal(crt_df_test, crt_df))
      assert_that(icc_crt_test$value == icc_crt$value)
  } else if (problem ==8){
    r_max_test = sqrt(icc_ravens$value * icc_crt$value)
    assert_that(r_max_test == r_max)
  } else if (problem==9){
    power_observed_test = pwr.r.test(n=NULL, r=ravens_crt_corr$estimate, sig.level=0.05, power=0.9)
    assert_that(power_observed$n == power_observed_test$n)
  } else if (problem == 10){
    power_minimum_test = pwr.r.test(n=NULL, r=ravens_crt_corr$conf.int[1], sig.level=0.05, power=0.9)
    assert_that(power_minimum_test$n == power_minimum$n)
  } else if (problem == 11){
    cor_values_test = seq(.1, .8, .1)
    power_levels_test = c(.8, .9)
    assert_that(all.equal(cor_values, cor_values_test))
    assert_that(all.equal(power_levels, power_levels_test))

    results_test = c()

    for (r in cor_values){
      for (p in power_levels){
        power_result = pwr.r.test(n=NULL, r=r, sig.level=0.05, power=p)
        results_test = rbind(results_test,
                        c(r,p,round(power_result$n)))
      }
    }

    results_df_test = data.frame(results_test)
    names(results_df_test) = c('r', 'power', 'n')
    results_df_test$power = factor(results_df_test$power)
    assert_that(all.equal(results_df, results_df_test))
  } else if (problem == 12){
    power_plot_test = ggplot(results_df, aes(r, n, group=power, color=power)) + 
      geom_line() + 
      xlab('correlation') + 
      ylab('required sample size')
    assert_that(all.equal(power_plot_test$data, power_plot$data))
  }
  print('good job!')
}