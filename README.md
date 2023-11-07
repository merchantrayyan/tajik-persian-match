# tajik-persian-match
In this module, brute force “transliteration” of consonants is used to compare
and match Tajik-Persian text strings:

```py
pip import match

>>>tg = 'Фориғ зи умеди раҳмату бими азоб'
>>>fa = 'فارغ ز امید رحمت و بیم عذاب'
>>>matched = match.ParallelText(match.match_words(tg, fa))
>>>matched
--------------------------------------------
Фориғ |‎ зи |‎ умеди |‎ раҳмату |‎ бими |‎ азоб                  
فارغ  |‎ ز  |‎ امید  |‎ رحمت و  |‎ بیم  |‎ عذاب  
 1.0  |‎1.0 |‎  1.0  |‎   1.0   |‎ 1.0  |‎ 1.0
--------------------------------------------
```
