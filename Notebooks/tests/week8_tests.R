library(assertthat)
check_values <- function(problem, seed=123456){
  if (problem == 1){

    generate_data_test = function(effect_size, pre_mean=50, pre_sd=5,  noise_sd=1, 
        n_per_group=500){
        df = data.frame(condition = rep(c('control', 'treatment'), each=n_per_group)) %>%
            mutate(
                # randomize the condition order and convert it to a factor variable
                condition = factor(sample(condition)),
                # create a true pre-treatment score for each person
                # by sampling from a random normal distribution
                # with mean = pre_mean and sd = pre_sd and n = 2 * n_per_group
                true_pre = rnorm(n = 2*n_per_group, mean=pre_mean, sd=pre_sd),
                # create an observed pre-treatment score by adding noise to the true score
                # noise should have mean = 0 and sd = noise_sd and n = 2 * n_per_group
                pre = true_pre + rnorm(2*n_per_group, mean=0, sd=noise_sd),
                # compute the treatment effect, which is defined by the effect size
                # this should only be nonzero for people in the treatment group
                # whose true pre score is above pre_mean
                treatment_effect = effect_size * (condition == 'treatment') * (true_pre > pre_mean),
                # create the post score, which is defined as the true pre score, plus random noise,
                # plus the treatment effect
                # the noise should have mean = 0 and sd = noise_sd and n = 2 * n_per_group
                post = true_pre + rnorm(2*n_per_group, mean=0, sd=noise_sd) + treatment_effect)

        return(df)
        }

    set.seed(seed)  # set the random seed for reproducibility

    effect_size=-0.5
    df_test = generate_data_test(effect_size)
    assert_that(all.equal(df, df_test))
  } else if (problem ==2) {
        attrition_test = function(df, attrition_rate=0.5, attrition_min=0.5){

            # find the cutoff for attrition due to high headache levels
            # using quantile()
            cutoff = quantile(df$true_pre, 1 - attrition_min) #only people in the top attrition_min drop out

            # create the attrited data frame
            df_attrition = df %>%
              mutate(
                # create a random attrition score for each person by sampling from a uniform distribution, with n = nrow(df)
                attrition_score=runif(n=nrow(df)),
                # create a new variable, attrition, which is TRUE if the true_pre score is above the cutoff
                # and the attrition score is less than the attrition rate
                attrition=(true_pre > cutoff & attrition_score < attrition_rate))
            return(df_attrition)
        }
        set.seed(seed)  # set the random seed for reproducibility
        effect_size=-0.5
        df_test = generate_data(effect_size)
        set.seed(seed)  # set the random seed for reproducibility
        df_attrition_test = attrition_test(df_test)
        assert_that(all.equal(df_attrition, df_attrition_test))
  } else if (problem ==3){
    # create a function to estimate the effect of attrition
        estimate_effect_of_attrition_test = function(df, df_attrition){

            # estimate the linear model, with post as the dependent variable and pre and condition
            # as the independent variables
            lm_result = lm(post ~ pre + condition, data=df)

            # create the summary of the model
            s = summary(lm_result)

            # estimate the model with attrition
            lm_result_attrition = lm(post ~ pre + condition, data=df_attrition %>% dplyr::filter(!attrition))

            # create the summary of the model with attrition
            s_attrition = summary(lm_result_attrition)

            
            result_df = data.frame(estimate_orig = s$coefficients[3,1],
                      p_orig = s$coefficients[3,4],
                      estimate_attrition = s_attrition$coefficients[3,1], 
                      p_attrition = s_attrition$coefficients[3,4])

            return(result_df)
        }
        set.seed(seed)  # set the random seed for reproducibility
        effect_size=-0.5
        df_test = generate_data(effect_size)
        set.seed(seed)  # set the random seed for reproducibility
        df_attrition_test = attrition(df_test)
        result_df_test = estimate_effect_of_attrition_test(df_test, df_attrition_test)
        assert_that(all.equal(result_df_test, result_df))


  } else if (problem == 4){
      set.seed(seed)

      results_df_test = c()

      for (i in 1:1000){
          df = generate_data(effect_size)
          df_attrition = attrition(df)
          results_df_test = rbind(results_df_test, estimate_effect_of_attrition(df, df_attrition))
      }
      assert_that(all.equal(results_df, results_df_test))

  } else if (problem ==5){
    results_sig_df_test = results_df %>%
        mutate(sigp_orig = p_orig < 0.05,
              sigp_attrition = p_attrition < 0.05) %>%
        dplyr::select(-c(p_orig, p_attrition))


    summary_df_test = results_sig_df_test %>%
          summarize_all(mean)
    assert_that(all.equal(summary_df,summary_df_test))
    assert_that(all.equal(results_sig_df, results_sig_df_test))
   } else if (problem == 6){
      set.seed(seed)
      sample_df_test = data.frame(group=rep(c('A', 'B'), group_size)) %>%
      mutate(
          # create the score for each individual by sampling from a normal distribution with mean = group_mean,
          # sd = noise_sd, and n = 2 * group_size
          # then add the group effect for members of group B, which is defined by group_effect_size * noise_sd
          # since the effect size is defined in standard deviation units
          score = rnorm(2*group_size, mean=group_mean, sd=noise_sd) + (group == 'B')*group_effect_size*noise_sd,
          # convert the group variable to a factor
          group = factor(group)
      )
      assert_that(all.equal(sample_df, sample_df_test))
   } else if (problem ==7){
        sample_summary_test = sample_df %>%
            group_by(group) %>%
            summarize(mean_score = mean(score),
                    sd_score = sd(score),
                    n = n())
        assert_that(all.equal(sample_summary, sample_summary_test))
   } else if (problem ==8){
    lr_model_test = glm(group ~ score, data=sample_df, family=binomial)
    assert_that(all.equal(lr_model$coefficients, lr_model_test$coefficients))
   } else if (problem==9){
    p_group_test = predict(lr_model, type='response')
    assert_that(all.equal(p_group, p_group_test))

   } else if (problem == 10){
    cutoff = 0.5 
    predicted_group_test = ifelse(p_group > cutoff, 'predB', 'predA')
    assert_that(all.equal(predicted_group, predicted_group_test))
   } else if (problem == 11){
    confusion_matrix_test = table(predicted_group, sample_df$group)
    assert_that(all.equal(confusion_matrix, confusion_matrix_test))

   }else if (problem == 12){
    accuracy_A_test = confusion_matrix[1,1] / sum(confusion_matrix[,1])
    accuracy_B_test = confusion_matrix[2,2] / sum(confusion_matrix[,2])
    mean_accuracy = (accuracy_A_test + accuracy_B_test) / 2
   }
  print('good job!')
}