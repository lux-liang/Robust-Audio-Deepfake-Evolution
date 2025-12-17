# Wav2Vec2 模型文件上传指南

## 目标目录
```
/root/aasist-main/pretrained/facebook/wav2vec2-xls-r-300m/
```

## 需要上传的文件
1. `config.json` (~2KB)
2. `pytorch_model.bin` (~1.27GB)

---

## 方法 1: 使用 scp (从本地 Windows/Mac 上传)

在您的本地电脑终端执行：

```bash
# 替换为您的实际服务器 IP 和用户名
scp config.json root@您的服务器IP:/root/aasist-main/pretrained/facebook/wav2vec2-xls-r-300m/
scp pytorch_model.bin root@您的服务器IP:/root/aasist-main/pretrained/facebook/wav2vec2-xls-r-300m/
```

**示例**（假设服务器 IP 是 `192.168.1.100`）:
```bash
scp config.json root@192.168.1.100:/root/aasist-main/pretrained/facebook/wav2vec2-xls-r-300m/
scp pytorch_model.bin root@192.168.1.100:/root/aasist-main/pretrained/facebook/wav2vec2-xls-r-300m/
```

---

## 方法 2: 使用 WinSCP / FileZilla (图形界面，推荐大文件)

1. 打开 **WinSCP** 或 **FileZilla**
2. 连接到服务器
3. 导航到: `/root/aasist-main/pretrained/facebook/wav2vec2-xls-r-300m/`
4. 拖拽上传两个文件

---

## 方法 3: 如果文件已经在服务器其他位置

```bash
# 假设文件在 /tmp/ 目录
cp /tmp/config.json /root/aasist-main/pretrained/facebook/wav2vec2-xls-r-300m/
cp /tmp/pytorch_model.bin /root/aasist-main/pretrained/facebook/wav2vec2-xls-r-300m/

# 或者使用 mv 移动
mv /tmp/config.json /root/aasist-main/pretrained/facebook/wav2vec2-xls-r-300m/
mv /tmp/pytorch_model.bin /root/aasist-main/pretrained/facebook/wav2vec2-xls-r-300m/
```

---

## 方法 4: 使用 rsync (支持断点续传，推荐大文件)

```bash
# 从本地同步到服务器（支持断点续传）
rsync -avz --progress config.json root@您的服务器IP:/root/aasist-main/pretrained/facebook/wav2vec2-xls-r-300m/
rsync -avz --progress pytorch_model.bin root@您的服务器IP:/root/aasist-main/pretrained/facebook/wav2vec2-xls-r-300m/
```

---

## 上传后验证

在服务器上执行以下命令验证：

```bash
cd /root/aasist-main/pretrained/facebook/wav2vec2-xls-r-300m/
ls -lh
```

**应该看到:**
```
-rw-r--r-- 1 root root   2.0K Nov 23 XX:XX config.json
-rw-r--r-- 1 root root   1.3G Nov 23 XX:XX pytorch_model.bin
```

---

## 快速检查脚本

上传完成后，运行以下命令检查：

```bash
cd /root/aasist-main && python3 << 'EOF'
import os
model_path = "./pretrained/facebook/wav2vec2-xls-r-300m"
required_files = ['config.json', 'pytorch_model.bin']

print("=" * 60)
print("文件检查:")
print("=" * 60)

all_ok = True
for f in required_files:
    file_path = os.path.join(model_path, f)
    if os.path.exists(file_path):
        size = os.path.getsize(file_path) / (1024**2)  # MB
        print(f"✅ {f}: 存在 ({size:.2f} MB)")
    else:
        print(f"❌ {f}: 缺失")
        all_ok = False

print("=" * 60)
if all_ok:
    print("✅ 所有文件已就绪，可以开始训练！")
else:
    print("❌ 还有文件缺失，请检查上传")
print("=" * 60)
EOF
```


