---
layout: post
title: "在 GitHub 上创建博客"
author: "Juewen Peng"
tags: Tutorial
excerpt_separator: <!--more-->
---

本文详细介绍了如何在 GitHub 上创建博客，应用 Jekyll 主题，以及在本地调试。<!--more--> 

主要参考 [https://www.cnblogs.com/sqchen/p/10757927.html](https://www.cnblogs.com/sqchen/p/10757927.html)。

<br>

## 1. 环境准备
安装 Git 并关联远程 GitHub 仓库，可参考 [https://www.cnblogs.com/lifexy/p/8353040.html](https://www.cnblogs.com/lifexy/p/8353040.html)。

<br>

## 2. 创建博客
### 2.1. 新建仓库
在 GitHub 上新建仓库，仓库名可以设成用户名（大小写都行）+固定后缀（.github.io）的形式，当作个人主页，或者设成其他任意名称。

<!-- ![img]({{site.baseurl}}/images/2022-06-22-blog-on-github/1.png){:width="60%"} -->

- **用户名+固定后缀**：进入 Settings -> Pages，应该可以看到以下格式的网页链接，可直接通过该链接访问博客。
<br>
[https://YOUR_USERNAME.github.io/](https://YOUR_USERNAME.github.io/)

- **其他任意名称**：进入 Settings -> Pages，将 Source 从 None 改为 Branch: main，并保存，此时应该会出现以下格式的网页链接。
<br>
[https://YOUR_USERNAME.github.io/YOUR_REPO_NAME/](https://YOUR_USERNAME.github.io/YOUR_REPO_NAME/)

### 2.2. 本地克隆
为了存放远程仓库，在本地创建一个文件夹，右键文件夹选择 `git bash here`，在当前目录打开 `git bash` 程序，执行如下程序，将远程仓库克隆到本地：

{% highlight bash %}
git clone https://github.com/JuewenPeng/Blog.git
{% endhighlight bash %}

<br>

## 3. 更换主题
可以在 GitHub 上或 [Jekyll Themes](http://jekyllthemes.org/page5/) 上选择自己喜欢的主题用来装饰博客界面，本博客的主题来源于 [Tale](http://jekyllthemes.org/themes/tale/)。下载完主题压缩包后把所有文件解压并移入到本地仓库的文件夹下（若提示文件已存在可直接覆盖）。

<br>

## 4. 本地搭建 Jekyll 环境
Github Page 本身是支持 Jekyll 环境的，但考虑到从远程 push 文件到网站更新界面需要较长时间，最好能在本地先完成界面的预览和调试，在确认没有问题之后再上传至 GitHub。

### 4.1. 安装 Ruby
进入 Ruby 软件的[主页](https://rubyinstaller.org/downloads/)，安装下载软件。

### 4.2. 在 Git Bash 中执行命令
**NOTE**：前两步一劳永逸

(1) 由于外网原因，换源，参考 [https://gems.ruby-china.com/](https://gems.ruby-china.com/)。
{% highlight bash %}
$ gem sources --add https://gems.ruby-china.com/ --remove https://rubygems.org/
$ gem sources -l
https://gems.ruby-china.com
# 确保只有 gems.ruby-china.com
{% endhighlight bash %}

(2) 安装 Jekyll
{% highlight bash %}
$ gem install jekyll
{% endhighlight bash %}

(3) 进入仓库文件夹中
{% highlight bash %}
$ cd Blog
{% endhighlight bash %}

(4) 更新并安装项目依赖包
{% highlight bash %}
$ bundle config mirror.https://rubygems.org https://gems.ruby-china.com
$ bundle update
$ bundle install
{% endhighlight %}

(5) 添加 WebRick 环境（Ruby 3.0.0 以上不会再自带 WebRick, 需要手动添加到环境里）
{% highlight bash %}
$ bundle add webrick
{% endhighlight bash %}

(6) 开启 Jekyll 服务
{% highlight bash %}
$ bundle exec jekyll serve
{% endhighlight bash %}
&nbsp;&nbsp;&nbsp;&nbsp;或者
{% highlight bash %}
$ jekyll serve
{% endhighlight bash %}

若上述代码均运行成功，应该能在 Git Bash 中看到 [http://127.0.0.1:4000](http://127.0.0.1:4000)，输入该网址，若界面呈现效果与主题 demo 相同，说明主题应用成功。若界面空白，不用担心，继续执行下一步。

<br>

## 5. 修改主题
进入 `_config.yml` 文件，将相关信息修改为自己的。记得将 `baseurl` 修改为自己的仓库子目录，本文中将其修改为 `"/blog"`。若创建仓库时采用第一种直接用户名的方式，此处修改为空字符串 `""`。重新开启 Jekyll 服务，并打开 [http://127.0.0.1:4000](http://127.0.0.1:4000)，查看主题样式。

<br>

## 6. 发布博客
在仓库文件夹下，进入 `_posts` 目录，所有的文章都必须放在 `_posts` 文件夹下。可参考 [Jekyll-目录结构](http://jekyllcn.com/docs/structure/) 查看各个文件夹的功能。

对于常用的 markdown 文章，按照如下格式创建 md 文件。
{% highlight bash %}
yyyy-MM-dd-filename.md
{% endhighlight bash %}

将编辑好的 md 文件 push 到远程仓库，完成更新。

{% highlight bash %}
$ git add .
$ git commit -m "upload"
$ git push origin master
{% endhighlight bash %}