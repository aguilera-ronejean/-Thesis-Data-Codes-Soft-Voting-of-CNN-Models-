function stop = savetrainingplot(info)
stop = false;
if info.State == "done"
    x=fix(clock);
    currentfig = findall(groot,'Type','Figure');
    saveas(currentfig,strcat('training ',num2str(x(5))), 'png')
     
end
end