syntax on
" colorscheme monochrome 
set number
set ruler

set nocompatible              " be iMproved, required
filetype off                  " required

" set the runtime path to include Vundle and initialize
set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()
" alternatively, pass a path where Vundle should install plugins
"call vundle#begin('~/some/path/here')

" let Vundle manage Vundle, required
Plugin 'fyquah95/vim-monochrome'
Plugin 'gmarik/Vundle.vim'
Plugin 'kien/ctrlp.vim'
Plugin 'vim-scripts/paredit.vim'
Plugin 'Valloric/YouCompleteMe'

" All of your Plugins must be added before the following line
call vundle#end()            " required
filetype plugin indent on    " required
" To ignore plugin indent changes, instead use:
"filetype plugin on
"
" Brief help
" :PluginList       - lists configured plugins
" :PluginInstall    - installs plugins; append `!` to update or just :PluginUpdate
" :PluginSearch foo - searches for foo; append `!` to refresh local cache
" :PluginClean      - confirms removal of unused plugins; append `!` to auto-approve removal
"
" see :h vundle for more details or wiki for FAQ
" Put your non-Plugin stuff after this line

set numberwidth=4
set shiftwidth=4
set softtabstop=4
set number
set expandtab
set smartindent
set smarttab
set autoindent
set ruler
let mapleader=","
nnoremap <leader>" viw<esc>a"<esc>hbi"<esc>lel
inoremap {<CR> {<CR>}<ESC>ko
inoremap jk <ESC>
vnoremap ," <ESC>`<i"<ESC>`>la"<ESC>l
vnoremap ,[ <ESC>`<i[<ESC>`>la]<ESC>l
set backspace=indent,eol,start
highlight Normal ctermfg=grey ctermbg=black

syntax on
set incsearch
" colorscheme monochrome 

" window stuff
nnoremap <leader>sw<left>  :topleft  vnew<CR>
nnoremap <leader>sw<right> :botright vnew<CR>
nnoremap <leader>sw<up>    :topleft  new<CR>
nnoremap <leader>sw<down>  :botright new<CR>

" Detect openCL File extension
au BufRead,BufNewFile *.cl set filetype=opencl
let g:ycm_global_ycm_extra_conf = '$HOME/.ycm_extra_conf.py'
let g:ycm_confirm_extra_conf = 0
au BufRead,BufNewFile *.maxj set filetype=java

let g:ctrlp_custom_ignore = {
            \ 'dir':  'resource_benchmark\|build/resource_bench_\|fpgaConvNetMaxeler_MAIA_DFE_SIM\|fpgaConvNetMaxeler_MAIA_DFE\|build' ,
            \ 'file': '\.class$\|\.o$\|\.log$\|\.dot$',
            \ }
