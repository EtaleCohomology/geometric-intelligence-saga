# s10 2038年問題と、時刻の決まり（第4章「時刻の決まり」、第6章「1901年」）
from datetime import datetime, timedelta, timezone

UTC = timezone.utc
epoch = datetime(1970, 1, 1, tzinfo=UTC)
max32 = 2**31 - 1
min32 = -2**31
print("符号付き32ビットの最大値:", f"{max32:,}", "秒 →", epoch + timedelta(seconds=max32))
print("次の１秒で跳ぶ先:", f"{min32:,}", "秒 →", epoch + timedelta(seconds=min32))


def wrap32(n):
    """符号付き32ビットの整数に押し込む（あふれると負に跳ぶ）"""
    return (n + 2**31) % 2**32 - 2**31


night = datetime(2038, 9, 21, 2, 7, tzinfo=UTC)
true_sec = int((night - epoch).total_seconds())
shown = epoch + timedelta(seconds=wrap32(true_sec))
print("事件の夜（2038-09-21 02:07 UTC）を32ビットで持つ機械が表す日付:", shown.date())
print("戻った年数:", round((night - shown).days / 365.2425, 1), "年")

# GPS の時刻と協定世界時の差（うるう秒は GPS に入らない）
# 2017年1月1日の時点で 18 秒。最後のうるう秒は 2016年12月31日。
gps_minus_utc = 18
gps_record = datetime(2038, 9, 21, 2, 7, 18, tzinfo=UTC)
print("GPS の時刻 02:07:18 を協定世界時に直すと:", (gps_record - timedelta(seconds=gps_minus_utc)).time(),
      "（記録の時点の公表値で確認すること）")

# １秒刻みの記録の揺れ：0〜1 秒にむらなく散らばる値の標準偏差
import math
print("１秒刻みの揺れの標準偏差 1/√12 =", round(1 / math.sqrt(12), 3), "秒")
