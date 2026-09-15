# assets —— 静态资源目录

> 本项目的前端静态资源（头像 / 图标）。由两个后端服务挂载到 `/assets` 路径提供访问，
> 例如：`<img src="/assets/avatars/bot/assistant.svg">`。

## 目录结构

```
assets/
├── avatars/                    头像
│   ├── bot/                    助手（听书智库）头像
│   │   ├── assistant.svg       主用：书本 + 声波（琥珀金→胡桃木渐变）
│   │   └── assistant-book.svg  备选：纯书本
│   └── user/                   用户默认头像（三选一）
│       ├── default.svg         经典人形剪影
│       ├── illustration.svg    扁平插画人物
│       └── initial.svg         木色底 + 衬线「我」字
└── icons/                      UI 图标（线性风格，颜色用 currentColor，可随主题）
    ├── book.svg                书本
    ├── book-audio.svg          书本 + 声波（品牌标记）
    ├── mic.svg                 麦克风（录音）
    ├── folder-upload.svg       文件夹上传
    ├── user.svg                用户
    └── send.svg                发送
```

## 使用约定

- **配色**：与前端设计语言一致（琥珀金 `#D9A441` / 胡桃木 `#6E4E33` / 米白 `#FBF6EE` / 暖米 `#F1E7D9`）。
- **尺寸**：`avatars/` 为 64×64 画布（圆角 18），`icons/` 为 24×24 画布。
- **图标颜色**：`icons/*.svg` 未写死颜色，使用 `stroke="currentColor"`，放进 HTML 后由 CSS 的 `color` 控制。
- **新增资源**：按用途放进对应子目录（头像→`avatars/`，UI 图标→`icons/`，插图/照片→新增 `images/`）。
