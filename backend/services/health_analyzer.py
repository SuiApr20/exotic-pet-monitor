"""
健康分析服务
基于行为统计数据，评估荷兰猪健康状态
"""
from datetime import datetime
from backend.config import HEALTH_THRESHOLDS


class HealthAnalyzer:
    """健康评分引擎"""

    def __init__(self):
        self.thresholds = HEALTH_THRESHOLDS

    def assess(self, stats: dict, observation_hours: float) -> dict:
        """
        基于统计数据评估健康

        参数:
            stats: ROI引擎统计 {total_eating_sec, total_drinking_sec, eating_events, drinking_events}
            observation_hours: 观察时长（小时）
        """
        score = 100
        alerts = []
        details = []

        total_eating = stats.get("total_eating_sec", 0)
        total_drinking = stats.get("total_drinking_sec", 0)
        eating_events = stats.get("eating_events", 0)
        drinking_events = stats.get("drinking_events", 0)

        # 每小时平均进食时间
        eating_per_hour = total_eating / observation_hours / 60 if observation_hours > 0 else 0

        # 每小时平均饮水次数
        drinking_per_hour = drinking_events / observation_hours if observation_hours > 0 else 0

        details.append({
            "metric": "进食时间/小时",
            "value": f"{eating_per_hour:.1f} 分钟",
            "normal": "荷兰猪全天持续啃草，每小时应有数次进食行为"
        })
        details.append({
            "metric": "饮水次数/小时",
            "value": f"{drinking_per_hour:.1f} 次",
            "normal": "饮水频率因个体而异，但应保持一定规律"
        })

        # 进食评估
        if observation_hours >= 2:
            if eating_per_hour < 1:  # 每小时进食不到1分钟
                score -= 30
                alerts.append({
                    "severity": "high",
                    "message": "进食时间严重不足，可能存在消化问题或牙齿异常",
                    "suggestion": "检查食盆是否有充足粮食，观察牙齿是否过长"
                })
            elif eating_per_hour < 3:
                score -= 15
                alerts.append({
                    "severity": "medium",
                    "message": "进食频率偏低，请持续观察",
                    "suggestion": "确保提摩西草不限量供应"
                })

        # 饮水评估
        if observation_hours >= 4 and drinking_per_hour < 0.5:
            score -= 20
            alerts.append({
                "severity": "high",
                "message": "饮水次数异常偏少，有脱水风险",
                "suggestion": "检查水瓶是否堵塞，提供新鲜水源"
            })

        # 综合评定
        if score >= 80:
            status = "健康"
            color = "success"
            summary = "进食和饮水行为正常，宠物状态良好"
        elif score >= 60:
            status = "需关注"
            color = "warning"
            summary = "行为指标偏低，建议加强观察"
        else:
            status = "警告"
            color = "danger"
            summary = "存在健康风险，建议尽快检查"

        # 无异常时也给出建议
        suggestions = []
        if total_eating < 60 * observation_hours:
            suggestions.append("建议增加粮食供应量")
        if drinking_events < 2 * observation_hours:
            suggestions.append("注意检查水壶是否正常出水")
        if not alerts:
            suggestions.append("暂未发现异常，继续正常喂养即可")

        return {
            "score": score,
            "status": status,
            "color": color,
            "summary": summary,
            "suggestions": suggestions,
            "alerts": alerts,
            "details": details,
            "thresholds": {
                "eating_low": self.thresholds["eating"]["alert_low_total_min"],
                "eating_no_visit_hours": self.thresholds["eating"]["alert_no_visit_hours"],
                "drinking_low": self.thresholds["drinking"]["alert_low_total_count"],
                "drinking_no_visit_hours": self.thresholds["drinking"]["alert_no_visit_hours"],
                "score_green": 80,
                "score_yellow": 60,
            },
            "observation_hours": observation_hours,
            "assessed_at": datetime.now().isoformat()
        }


health_analyzer = HealthAnalyzer()
