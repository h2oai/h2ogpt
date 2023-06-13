# Settings
$xdvipdfmx = "xdvipdfmx -z 6 -i dvipdfmx-unsafe.cfg -o %D %O %S";

# Workaround to allow pstricks transparency (https://github.com/overleaf/issues/issues/3449)
$dvipdf = "dvipdf -dNOSAFER -dALLOWPSTRANSPARENCY %O %S %D";

###############################
# Post processing of pdf file #
###############################

$compiling_cmd = "internal overleaf_pre_process %T %D";
$success_cmd = "internal overleaf_post_process %T %D";
$failure_cmd = $success_cmd;

my $ORIG_PDF_AGE;

sub overleaf_pre_process {
    my $source_file = $_[0];
    my $output_file = $_[1];

    # get age of existing pdf if present
    $ORIG_PDF_AGE = -M $output_file
}

sub overleaf_post_process {
    my $source_file = $_[0];
    my $output_file = $_[1];
    my $source_without_ext = $source_file =~ s/\.tex$//r;
    my $output_without_ext = $output_file =~ s/\.pdf$//r;

    # Look for a knitr concordance file
    my $concordance_file = "${source_without_ext}-concordance.tex";
    if (-e $concordance_file) {
        print "Patching synctex file for knitr...\n";
        system("patchSynctex.R", $source_without_ext, $output_without_ext);
    }

    # Return early if pdf file doesn't exist or wasn't updated
    my $NEW_PDF_AGE = -M $output_file;
    return if !defined($NEW_PDF_AGE);
    return if defined($ORIG_PDF_AGE) && $NEW_PDF_AGE == $ORIG_PDF_AGE;

    # Figure out where qpdf is
    $qpdf //= "/usr/bin/qpdf";
    $qpdf = $ENV{QPDF} if defined($ENV{QPDF}) && -x $ENV{QPDF};
    return if ! -x $qpdf;
    $qpdf_opts //= "--linearize --newline-before-endstream";
    $qpdf_opts = $ENV{QPDF_OPTS} if defined($ENV{QPDF_OPTS});

    # Run qpdf
    my $optimised_file = "${output_file}.opt";
    system($qpdf, split(' ', $qpdf_opts), $output_file, $optimised_file);
    $qpdf_exit_code = ($? >> 8);
    print "qpdf exit code=$qpdf_exit_code\n";

    # Replace the output file if qpdf was successful
    # qpdf returns 0 for success, 3 for warnings (output pdf still created)
    return if !($qpdf_exit_code == 0 || $qpdf_exit_code == 3);
    print "Renaming optimised file to $output_file\n";
    rename($optimised_file, $output_file);

    print "Extracting xref table for $output_file\n";
    my $xref_file = "${output_file}xref";
    system("$qpdf --show-xref ${output_file} > ${xref_file}");
    $qpdf_xref_exit_code = ($? >> 8);
    print "qpdf --show-xref exit code=$qpdf_xref_exit_code\n";
}

##############
# Glossaries #
##############
add_cus_dep( 'glo', 'gls', 0, 'glo2gls' );
add_cus_dep( 'acn', 'acr', 0, 'glo2gls');  # from Overleaf v1
sub glo2gls {
    system("makeglossaries $_[0]");
}

#############
# makeindex #
#############
@ist = glob("*.ist");
if (scalar(@ist) > 0) {
    $makeindex = "makeindex -s $ist[0] %O -o %D %S";
}

################
# nomenclature #
################
add_cus_dep("nlo", "nls", 0, "nlo2nls");
sub nlo2nls {
        system("makeindex $_[0].nlo -s nomencl.ist -o $_[0].nls -t $_[0].nlg");
}

#########
# Knitr #
#########
add_cus_dep( 'Rtex', 'tex', 0, 'do_knitr');
sub do_knitr {
    Run_subst(qq{Rscript -e '
        library("knitr");
        opts_knit\$set(concordance=T);
        knitr::knit(%S, output=%D);
        '}
    );
}

##########
# feynmf #
##########
push(@file_not_found, '^feynmf: Files .* and (.*) not found:$');
add_cus_dep("mf", "tfm", 0, "mf_to_tfm");
sub mf_to_tfm { system("mf '\\mode:=laserjet; input $_[0]'"); }

push(@file_not_found, '^feynmf: Label file (.*) not found:$');
add_cus_dep("mf", "t1", 0, "mf_to_label1");
sub mf_to_label1 { system("mf '\\mode:=laserjet; input $_[0]' && touch $_[0].t1"); }
add_cus_dep("mf", "t2", 0, "mf_to_label2");
sub mf_to_label2 { system("mf '\\mode:=laserjet; input $_[0]' && touch $_[0].t2"); }
add_cus_dep("mf", "t3", 0, "mf_to_label3");
sub mf_to_label3 { system("mf '\\mode:=laserjet; input $_[0]' && touch $_[0].t3"); }
add_cus_dep("mf", "t4", 0, "mf_to_label4");
sub mf_to_label4 { system("mf '\\mode:=laserjet; input $_[0]' && touch $_[0].t4"); }
add_cus_dep("mf", "t5", 0, "mf_to_label5");
sub mf_to_label5 { system("mf '\\mode:=laserjet; input $_[0]' && touch $_[0].t5"); }
add_cus_dep("mf", "t6", 0, "mf_to_label6");
sub mf_to_label6 { system("mf '\\mode:=laserjet; input $_[0]' && touch $_[0].t6"); }
add_cus_dep("mf", "t7", 0, "mf_to_label7");
sub mf_to_label7 { system("mf '\\mode:=laserjet; input $_[0]' && touch $_[0].t7"); }
add_cus_dep("mf", "t8", 0, "mf_to_label8");
sub mf_to_label8 { system("mf '\\mode:=laserjet; input $_[0]' && touch $_[0].t8"); }
add_cus_dep("mf", "t9", 0, "mf_to_label9");
sub mf_to_label9 { system("mf '\\mode:=laserjet; input $_[0]' && touch $_[0].t9"); }

##########
# feynmp #
##########
push(@file_not_found, '^dvipdf: Could not find figure file (.*); continuing.$');
add_cus_dep("mp", "1", 0, "mp_to_eps");
sub mp_to_eps {
    system("mpost $_[0]");
    return 0;
}

#############
# asymptote #
#############

sub asy {return system("asy \"$_[0]\"");}
add_cus_dep("asy","eps",0,"asy");
add_cus_dep("asy","pdf",0,"asy");
add_cus_dep("asy","tex",0,"asy");

#############
# metapost  #  # from Overleaf v1
#############
add_cus_dep('mp', '1', 0, 'mpost');
sub mpost {
    my $file = $_[0];
    my ($name, $path) = fileparse($file);
    pushd($path);
    my $return = system "mpost $name";
    popd();
    return $return;
}

##########
# chktex #
##########
unlink 'output.chktex' if -f 'output.chktex';
if (defined $ENV{'CHKTEX_OPTIONS'}) {
    use File::Basename;
    use Cwd;

    # identify the main file
    my $target = $ARGV[-1];
    my $file = basename($target);

    if ($file =~ /\.tex$/) {
        # change directory for a limited scope
        my $orig_dir = cwd();
        my $subdir = dirname($target);
        chdir($subdir);
        # run chktex on main file
        $status = system("/usr/local/bin/run-chktex.sh", $orig_dir, $file);
        # go back to original directory
        chdir($orig_dir);

        # in VALIDATE mode we always exit after running chktex
        # otherwise we exit if EXIT_ON_ERROR is set

        if ($ENV{'CHKTEX_EXIT_ON_ERROR'} || $ENV{'CHKTEX_VALIDATE'}) {
            # chktex doesn't let us access the error info via exit status
            # so look through the output
            open(my $fh, "<", "output.chktex");
            my $errors = 0;
            {
                local $/ = "\n";
                while(<$fh>) {
                    if (/^\S+:\d+:\d+: Error:/) {
                        $errors++;
                        print;
                    }
                }
            }
            close($fh);
            exit(1) if $errors > 0;
            exit(0) if $ENV{'CHKTEX_VALIDATE'};
        }
    }
}
