# 电竞8强淘汰赛晋级图 / E-Sports 8-Team Tournament Bracket

## 简介 (Introduction)

这是一个交互式的8强淘汰赛晋级图，专为电子竞技比赛设计。支持实时编辑队伍名称、输入比分、自动计算获胜者并晋级到下一轮。

An interactive 8-team knockout tournament bracket designed for e-sports competitions. Features real-time team name editing, score input, automatic winner calculation, and advancement to the next round.

## 如何使用 (How to Use)

### 在浏览器中打开 (Open in Browser)

直接在浏览器中打开 `esports-tournament-bracket.html` 文件即可使用。

Simply open the `esports-tournament-bracket.html` file in your web browser.

### 功能说明 (Features)

#### 1. 编辑队伍名称 (Edit Team Names)
- 点击任意队伍名称即可编辑
- Click on any team name to edit it

#### 2. 输入比分 (Enter Scores)
- 点击分数框输入比赛分数
- Click on the score boxes to enter match scores
- 分数较高的队伍自动高亮显示（绿色背景）
- The team with the higher score is automatically highlighted (green background)

#### 3. 自动晋级 (Automatic Advancement)
- 输入比分后，获胜队伍自动晋级到下一轮
- After entering scores, winning teams automatically advance to the next round
- 四分之一决赛 → 半决赛 → 决赛 → 冠军
- Quarterfinals → Semifinals → Finals → Champion

#### 4. 重置赛程 (Reset Tournament)
- 点击"重置赛程"按钮可以清空所有数据
- Click the "Reset Tournament" button to clear all data

#### 5. 打印赛程 (Print Bracket)
- 点击"打印赛程"按钮可以打印当前赛程表
- Click the "Print Bracket" button to print the current bracket

#### 6. 数据持久化 (Data Persistence)
- 所有更改自动保存到浏览器的 localStorage
- All changes are automatically saved to the browser's localStorage
- 刷新页面后数据仍然保留
- Data persists after page refresh

## 技术特点 (Technical Features)

- ✅ 纯 HTML/CSS/JavaScript，无需任何依赖
- ✅ Pure HTML/CSS/JavaScript, no dependencies required
- ✅ 响应式设计，适配移动设备和桌面
- ✅ Responsive design, works on mobile and desktop
- ✅ 现代化的渐变配色和动画效果
- ✅ Modern gradient colors and animation effects
- ✅ 中英文双语支持
- ✅ Bilingual support (Chinese/English)

## 赛制说明 (Tournament Format)

```
四分之一决赛 (Quarterfinals) - 8支队伍
        ↓
半决赛 (Semifinals) - 4支队伍
        ↓
决赛 (Finals) - 2支队伍
        ↓
冠军 (Champion) - 1支队伍
```

## 浏览器兼容性 (Browser Compatibility)

- ✅ Chrome/Edge (推荐 Recommended)
- ✅ Firefox
- ✅ Safari
- ✅ 其他现代浏览器 (Other modern browsers)

## 许可 (License)

本项目使用 MIT 许可证开源。

This project is open source under the MIT License.
