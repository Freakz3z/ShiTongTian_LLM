# DeepSeek-Recipie-Food
<img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/made with unsloth.png" height="50" align="center" />
模型已上传至HuggingFace: https://huggingface.co/Freakzzz/model

# 数据集：
[AkashPS11/recipes_data_food.com](https://huggingface.co/datasets/AkashPS11/recipes_data_food.com)
# 训练：
例：
```
Prompt:
[{'content': '\nRespond in the following format:\n<reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>\n', 'role': 'system'}, {'content': 'Low-Fat Berry Blue Frozen Dessert', 'role': 'user'}]
Answer:
<reasoning>
Recipe for Low-Fat Berry Blue Frozen Dessert
</reasoning>
<answer>
<recipe>
<name>
Low-Fat Berry Blue Frozen Dessert
</name>
<ingredients>
c("4" c("blueberries", "1/4" "granulated sugar", "1" "vanilla yogurt", "1") "lemon juice")
</ingredients>
<instructions>
c("Toss 2 cups berries with sugar.", "Let stand for 45 minutes, stirring occasionally.", "Transfer berry-sugar mixture to food processor.", "Add yogurt and process until smooth.", "Strain through fine sieve. Pour into baking pan (or transfer to ice cream maker and process according to manufacturers' directions). Freeze uncovered until edges are solid but centre is soft.  Transfer to processor and blend until smooth again.", "Return to pan and freeze until edges are solid.", "Transfer to processor and blend until smooth again.", 
"Fold in remaining 2 cups of blueberries.", "Pour into plastic mold and freeze overnight. Let soften slightly to serve.")
</instructions>
<nutrition>
Calories: 170.9, Fat: 2.5g, Saturated Fat: 1.3g, Cholesterol: 8.0mg, Sodium: 29.8mg, Carbohydrates: 37.1g, Fiber: 3.6g, Sugar: 30.2g, Protein: 3.2g
</nutrition>
</recipe>
</answer>
```
# 回答
使用`使用中文回答：为我推荐一个食谱，并告诉我其中的营养含量`作为输入，对比无微调与训练30min后的结果：

不使用GRPO训练：
```
今天我给你推荐一个健康又美味的食谱：甘蓝奶酪饼（Kale Quiche）！

**甘蓝奶酪饼食谱**

**原料：**

*   甘蓝叶 2杯
*   奶酪 2杯
*   生鸡蛋 4个
*   蛋黄 1杯
*   面粉 1杯
*   精盐 1小匙
*   黑胡椒 1小匙
*   淡奶 1/2 杯

**做法：**

1.  预热烤箱至375度。
2.  甘蓝叶洗干净，切碎。
3.  在一个大碗中，将甘蓝叶、奶酪、鸡蛋、蛋黄、面粉、精盐和黑胡椒混合。
4.  混合后在一个抹有奶油的9英寸方形盘中倒入混合物。
5.  淡奶浇在上面。
6.  放入烤箱，烤约25-30分钟，直到表面金黄。
7.  取出后让它稍微冷却，切片享用。

**营养含量（每份约60克）**

*   蛋白质：15克
*   卡路里：220
*   膳食纤维：2克
*   蛋黄：10克
*   蛋白质：3克
*   钙：10%DV
*   维生素A：10%DV
*   维生素C：25%DV
*   维生素K：500%DV

这个甘蓝奶酪饼不仅美味，还富含维生素A、C和K，以及蛋白质和膳食纤维，是一个营养均衡的选择。
```
使用GRPO训练：
```
**清蒸凤爪**

**食材**

*   2个凤爪
*   2 cloves
*   2个生姜
*   2汤匙白葡萄酒
*   2汤匙生抽油
*   1汤匙酱油
*   1汤匙生抽
*   1汤匙糖
*   1/2茶匙盐
*   1/4茶匙白胡椒粉
*   2汤匙葱末
*   2汤匙姜末
*   2汤匙生姜末
*   1汤匙陈醋
*   2汤匙水
*   1/4杯清水

**做法**

1.  凤爪洗净，放入沸水中煮5分钟后捞出，去骨后切成小块
2.  在热锅中加油，爆香姜、生姜和葱末
3.  加入葡萄酒、生抽、酱油、生抽、糖、盐和白胡椒粉，翻炒至冒烟
4.  加入水和陈醋，烧开后倒入中铸锅，浸泡凤爪
5.  层锅加盖，高火烧开后转小火，用少量清水浸泡20分钟
6.  低火浸泡完毕后，撇去浮油，淋上葱末和生姜末

**营养含量**

*   每份凤爪（约100克）含有
    *   蛋白质：22克
    *   脂肪：10克
    *   碳水化合物
```
