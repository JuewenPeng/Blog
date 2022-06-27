---
layout: post
title: "在 GitHub 上创建博客"
author: "Juewen Peng"
comments: false
tags: [Tutorial, GitHub, Tale, Jekyll]
excerpt_separator: <!--more-->
sticky: false
hidden: false
katex: true
---

<!-- "highlight language" refer to https://github.com/rouge-ruby/rouge/wiki/List-of-supported-languages-and-lexers -->

本文详细介绍了如何在 GitHub 上创建博客，应用 Jekyll 主题，以及在本地调试。<!--more--> 

主要参考 [https://www.cnblogs.com/sqchen/p/10757927.html](https://www.cnblogs.com/sqchen/p/10757927.html)。

**2022/06/27 更新**：增加新章节 “KaTeX 加速 LaTeX 公式渲染”

---

<br>

## 1. 环境准备
安装 Git 并关联远程 GitHub 仓库，可参考 [https://www.cnblogs.com/lifexy/p/8353040.html](https://www.cnblogs.com/lifexy/p/8353040.html)。

<br>

## 2. 创建博客
### 2.1. 新建仓库
在 GitHub 上新建仓库，仓库名可以设成用户名（大小写都行）+固定后缀（.github.io）的形式，当作个人主页，或者设成其他任意名称。

<!-- ![img]({{site.baseurl}}/images/2022-06-22-blog-on-github/1.png){:width="60%"} -->

- **用户名+固定后缀**：进入 Settings -> Pages，应该能看到以下格式的网页链接，该链接即为博客网址。
<br>
[https://YOUR_USERNAME.github.io/](https://YOUR_USERNAME.github.io/)

- **其他任意名称**：进入 Settings -> Pages，将 Source 下方的 `None` 改为 `main` 并保存，此时应该会出现以下格式的网页链接，同样，该链接即为博客网址。
<br>
[https://YOUR_USERNAME.github.io/YOUR_REPO_NAME/](https://YOUR_USERNAME.github.io/YOUR_REPO_NAME/)

### 2.2. 本地克隆
为了存放远程仓库，在本地创建一个文件夹，右键文件夹选择 `git bash here`，在当前目录打开 `git bash` 程序，执行如下程序，将远程仓库克隆到本地：

{% highlight shell %}
$ git clone https://github.com/JuewenPeng/Blog.git
{% endhighlight %}

<br>

## 3. 更换主题
可以在 GitHub 上或 [Jekyll Themes](http://jekyllthemes.org/page5/) 上选择自己喜欢的主题用来装饰博客界面，本博客的主题来源于 [Tale](http://jekyllthemes.org/themes/tale/)。下载完主题压缩包后把所有文件解压并移入到本地仓库的文件夹下（同名文件可直接覆盖）。

<br>

## 4. 本地搭建 Jekyll 环境
Github Page 本身是支持 Jekyll 环境的，但考虑到从远程 push 文件到网站更新界面需要较长时间，最好能在本地先完成界面的预览和调试，在确认没有问题之后再上传至 GitHub。

### 4.1. 安装 Ruby
进入 Ruby 软件的[主页](https://rubyinstaller.org/downloads/)，安装下载软件。

### 4.2. 在 Git Bash 中执行命令
**NOTE**：前两步一劳永逸

(1) 由于外网原因，换源，参考 [https://gems.ruby-china.com/](https://gems.ruby-china.com/)。
{% highlight shell %}
$ gem sources --add https://gems.ruby-china.com/ --remove https://rubygems.org/
$ gem sources -l
https://gems.ruby-china.com
# 确保只有 gems.ruby-china.com
{% endhighlight %}

(2) 安装 Jekyll
{% highlight shell %}
$ gem install jekyll
{% endhighlight %}

(3) 进入仓库文件夹中
{% highlight shell %}
$ cd Blog
{% endhighlight %}

(4) 更新并安装项目依赖包
{% highlight shell %}
$ bundle config mirror.https://rubygems.org https://gems.ruby-china.com
$ bundle update
$ bundle install
{% endhighlight %}

(5) 添加 WebRick 环境（Ruby 3.0.0 以上不会再自带 WebRick, 需要手动添加到环境里）
{% highlight shell %}
$ bundle add webrick
{% endhighlight %}

(6) 开启 Jekyll 服务
{% highlight shell %}
$ bundle exec jekyll serve
{% endhighlight %}
&nbsp;&nbsp;&nbsp;&nbsp;或者
{% highlight shell %}
$ jekyll serve
{% endhighlight %}

若上述代码均运行成功，应该能在 Git Bash 中看到 [http://127.0.0.1:4000](http://127.0.0.1:4000)，输入该网址，若界面呈现效果与主题 demo 相同，说明主题应用成功。若界面空白，不用担心，继续执行下一步。

<br>

## 5. 修改主题
进入 `_config.yml` 文件，将相关信息修改为自己的。注意，若仓库名采用第一种**用户名+固定后缀**的方式，将 `baseurl` 改为空字符串 `""`；若采用第二种**任意名称**的方式，将 `baseurl` 改为仓库名，如本文应将其修改为 `"/blog"`。修改完相关信息后，按照上一小节的最后一步重新开启 Jekyll 服务，并打开网页 [http://127.0.0.1:4000](http://127.0.0.1:4000)，查看主题样式。

<br>

## 6. 发布博客
在仓库文件夹下，进入 `_posts` 目录，所有的文章都必须放在 `_posts` 文件夹下。可参考 [Jekyll-目录结构](http://jekyllcn.com/docs/structure/) 查看各个文件夹的功能。

对于常用的 markdown 文章，按照如下格式创建 md 文件。
{% highlight plaintext %}
yyyy-MM-dd-filename.md
{% endhighlight %}

将编辑好的 md 文件 push 到远程仓库，完成更新。

<br>

## 7. KaTeX 加速 LaTeX 公式渲染
Jekyll 默认使用 MathJax 来渲染公式，当界面公式较多时，页面加载会变慢。KaTeX 是一个支持在网页上显示 LaTeX 公式的 JavaScript 库，相较于 MathJax，KaTeX 渲染速度要快得多，可以通过 [KaTeX and MathJax Comparison Demo](http://www.intmath.com/cg5/KaTeX-mathjax-comparison.php) 比较一下两者的渲染速度。

具体的操作配置流程见 [How to LaTeX in Jekyll using KaTeX](https://www.xuningyang.com/blog/2021-01-11-katex-with-jekyll/)

若要在本地 Jekyll 环境上跑，在上述链接的 Step 2 中需要执行 Method 2。

原文中写到下面三个包只要安装一个就行，本人推荐安装第三个（第一个包安装不上，第二个包后续开启 Jekyll 服务报错）。
{% highlight shell %}
$ gem install therubyracer
$ gem install therubyrhino
$ gem install duktape (recommend)
{% endhighlight %}

另外，我们还需要通过下面的命令将安装的包手动添加到本地环境：
{% highlight shell %}
$ bundle add kramdown-math-katex
$ bundle add katex
$ bundle add execjs
$ bundle add duktape
{% endhighlight %}

测试（注意格式与 LaTeX 稍有不同）：

{% highlight markdown %}
行内公式: $$f(x) = \int_{-\infty}^\infty \hat f(\xi)\,e^{2 \pi i \xi x} \,d\xi$$

行间公式:

$$f(x) = \int_{-\infty}^\infty \hat f(\xi)\,e^{2 \pi i \xi x} \,d\xi$$
{% endhighlight %}

行内公式: $$f(x) = \int_{-\infty}^\infty \hat f(\xi)\,e^{2 \pi i \xi x} \,d\xi$$

行间公式:

$$f(x) = \int_{-\infty}^\infty \hat f(\xi)\,e^{2 \pi i \xi x} \,d\xi$$