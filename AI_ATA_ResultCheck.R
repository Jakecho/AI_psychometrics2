## ---------------------------
##
## Script name: <your_script_name>.R
##
## Purpose of script: <brief description>
##
## Author: Jake Cho, PhD
##
## Date Created: 2025-11-22
##
## Copyright (c) Meazure Learning, 2025
##
## Email: jcho@meazurelearning.com
##
## ---------------------------
##
## Notes:
##   - <add notes or dependencies>
##
## ---------------------------
library(openxlsx)
library(tidyverse)
setwd("C:/Users/JakeCho/Projects/Conferences/ICE Exchange2025Nov/")
f1 <- read.xlsx("AI_ATA_forms_assembly.xlsx", sheet = "Form_1")
f2 <- read.xlsx("AI_ATA_forms_assembly.xlsx", sheet = "Form_2")
f3 <- read.xlsx("AI_ATA_forms_assembly.xlsx", sheet = "Form_3")

#count the number of unique domians in each form
table(f1$domain)
table(f2$domain)
table(f3$domain)

mean(f1$rasch_b)
mean(f2$rasch_b)
mean(f3$rasch_b)

mean(f1$point_biserial)
mean(f2$point_biserial)
mean(f3$point_biserial) 

#find the common items across the three forms and between two forms
common_items_1_2 <- intersect(f1$item_id, f2$item_id)
common_items_1_3 <- intersect(f1$item_id, f3$item_id)
common_items_2_3 <- intersect(f2$item_id, f3$item_id)
common_items_all <- Reduce(intersect, list(f1$item_id, f2$item_id, f3$item_id))
#check the all four common items are identical
f1_common <- f1 %>% filter(item_id %in% common_items_all) %>% arrange(item_id)
f2_common <- f2 %>% filter(item_id %in% common_items_all) %>% arrange(item_id)
f3_common <- f3 %>% filter(item_id %in% common_items_all) %>% arrange(item_id)
all.equal(f1_common, f2_common)
all.equal(f1_common, f3_common)
all.equal(f2_common, f3_common)


# coutn the unique items in each form
unique_items_f1 <- setdiff(f1$item_id, union(f2$item_id, f3$item_id))
unique_items_f2 <- setdiff(f2$item_id, union(f1$item_id, f3$item_id))
unique_items_f3 <- setdiff(f3$item_id, union(f1$item_id, f2$item_id))

